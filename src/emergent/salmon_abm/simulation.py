"""Skeleton `simulation` class extracted from sockeye.py.

This minimal class preserves the original constructor signature and
provides `timestep`, `run`, and `close` methods so other code can import
and be migrated incrementally. The implementation is intentionally light
weight to remain testable without heavy environment data.
"""
import os
import tempfile
import h5py
import numpy as np
from typing import Optional
from emergent.salmon_abm import utils, io, pid, agents, hdf5_io


class simulation:
    def __init__(self, 
                 model_dir, 
                 model_name, 
                 crs, 
                 basin, 
                 water_temp, 
                 start_polygon,
                 env_files,
                 longitudinal_profile,
                 fish_length = None,
                 num_timesteps = 100, 
                 num_agents = 100,
                 use_gpu = False,
                 pid_tuning = False,
                 db_path: Optional[str] = None):
        self.model_dir = model_dir
        self.model_name = model_name
        self.crs = crs
        self.basin = basin
        self.water_temp = water_temp
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        # minimal state
        self.X = np.zeros(num_agents, dtype=np.float32)
        self.Y = np.zeros(num_agents, dtype=np.float32)
        self.dead = np.zeros(num_agents, dtype=np.int8)
        self.cumulative_time = 0.0
        self.env_files = env_files or []
        self.longitudinal_profile = longitudinal_profile

        # prepare simple RNG to keep behavior deterministic when seed used
        try:
            # agents may set RNG via self.rng if needed
            self.rng = np.random.default_rng()
        except Exception:
            self.rng = None
        # keep a pid controller if requested
        self.pid_controller = pid.PID_controller(num_agents) if pid_tuning else None
        
        # create in-memory arrays for agent attributes so agent generators can populate them
        self.sex = np.zeros(self.num_agents, dtype=np.int8)
        self.length = np.zeros(self.num_agents, dtype=np.float32)
        self.weight = np.zeros(self.num_agents, dtype=np.float32)
        self.body_depth = np.zeros(self.num_agents, dtype=np.float32)

        # create or open HDF5 database for simulation outputs (minimal structure)
        self._created_db_file = False
        if db_path:
            self.db_path = db_path
            self.db = h5py.File(self.db_path, "w")
        else:
            # Place temporary DB in repository `outputs/` to avoid OS temp permission issues
            repo_outputs = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs"))
            os.makedirs(repo_outputs, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix="sim_db_", suffix=".h5", dir=repo_outputs)
            os.close(fd)
            self.db_path = tmp_path
            self.db = h5py.File(self.db_path, "w")
            self._created_db_file = True

        # create static per-fish datasets via hdf5_io
        # create both legacy top-level datasets and namespaced agent_data for compatibility
        hdf5_io.write_dataset(self.db, "sex", np.zeros((self.num_agents,), dtype=np.int8), dtype="i1")
        hdf5_io.write_dataset(self.db, "length", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "weight", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "body_depth", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "agent_data/sex", np.zeros((self.num_agents,), dtype=np.int8), dtype="i1")
        hdf5_io.write_dataset(self.db, "agent_data/length", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "agent_data/weight", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "agent_data/body_depth", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")

        # environment masks/metadata placeholders
        hdf5_io.write_dataset(self.db, "too_shallow", np.zeros((1,), dtype=np.int8), dtype="i1")
        hdf5_io.write_dataset(self.db, "opt_wat_depth", np.zeros((1,), dtype=np.float32), dtype="f4")

        # simple position datasets (time-varying axes would be added by run)
        hdf5_io.write_dataset(self.db, "X", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")
        hdf5_io.write_dataset(self.db, "Y", np.zeros((self.num_agents,), dtype=np.float32), dtype="f4")

        # populate agent attributes using the agents module
        try:
            agents.sim_sex(self)
            agents.sim_length(self, fish_length)
            agents.sim_weight(self)
            agents.sim_body_depth(self)
        except Exception:
            # if agents fail, leave zeros but continue
            pass

        # write agent attributes into HDF5 static datasets (best-effort)
        try:
            hdf5_io.write_dataset(self.db, "agent_data/sex", self.sex)
            hdf5_io.write_dataset(self.db, "agent_data/length", self.length)
            hdf5_io.write_dataset(self.db, "agent_data/weight", self.weight)
            hdf5_io.write_dataset(self.db, "agent_data/body_depth", self.body_depth)
            # also write legacy top-level datasets for compatibility
            hdf5_io.write_dataset(self.db, "sex", self.sex)
            hdf5_io.write_dataset(self.db, "length", self.length)
            hdf5_io.write_dataset(self.db, "weight", self.weight)
            hdf5_io.write_dataset(self.db, "body_depth", self.body_depth)
            try:
                # ensure changes are persisted for h5py.File
                if hasattr(self.db, 'flush'):
                    self.db.flush()
            except Exception:
                pass
        except Exception:
            # non-fatal; continue
            pass

        # if env_files provided, try to import them using io.enviro_import (non-fatal)
        for ef in self.env_files:
            try:
                _ = io.enviro_import(ef)
            except Exception:
                continue

        # ensure minimal environment placeholders exist so downstream modules
        # that read environment/* will have something to sample in unit tests
        try:
            hdf5_io.create_environment_placeholders(self.db)
        except Exception:
            pass

    def timestep(self, t, dt, g=None, pid_controller=None):
        # Advance simple odometer and time
        self.cumulative_time += dt
        return True

    def run(self, model_name=None, n=1, dt=1.0, video=False, k_p=None, k_i=None, k_d=None):
        # Simple run loop that calls `timestep` n times
        for i in range(n):
            self.timestep(i, dt)
        return True

    def close(self):
        # close HDF5 and optionally remove temporary DB file if it was created internally
        try:
            if hasattr(self, "db") and self.db is not None:
                try:
                    self.db.close()
                except Exception:
                    pass
            if getattr(self, "_created_db_file", False):
                try:
                    os.remove(self.db_path)
                except Exception:
                    pass
        finally:
            return True


__all__ = ['simulation']
