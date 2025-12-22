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
import logging
from typing import Optional
from emergent.salmon_abm import utils, io, pid, agents, hdf5_io
from emergent.salmon_abm import movement as movement_mod, behavior as behavior_mod, fatigue as fatigue_mod


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
        # runtime state expected by extracted modules (safe defaults)
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()
        self.x_vel = np.zeros(self.num_agents, dtype=np.float32)
        self.y_vel = np.zeros(self.num_agents, dtype=np.float32)
        self.heading = np.zeros(self.num_agents, dtype=np.float32)
        self.sog = np.zeros(self.num_agents, dtype=np.float32)
        self.ideal_sog = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_Hz = np.zeros(self.num_agents, dtype=np.float32)
        self.Hz = np.zeros(self.num_agents, dtype=np.float32)
        self.thrust = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.drag = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.swim_behav = np.ones(self.num_agents, dtype=np.int8)
        self.battery = np.ones(self.num_agents, dtype=np.float32)
        self.recover_stopwatch = np.zeros(self.num_agents, dtype=np.float32)
        self.swim_speeds = np.zeros((self.num_agents, 5), dtype=np.float32)
        self.dist_per_bout = np.zeros(self.num_agents, dtype=np.float32)
        self.bout_dur = np.zeros(self.num_agents, dtype=np.float32)
        self.swim_mode = np.ones(self.num_agents, dtype=np.int8)
        self.max_s_U = np.repeat(2.77, self.num_agents)
        self.max_p_U = np.repeat(4.43, self.num_agents)
        self.a_p = np.repeat(0.0, self.num_agents)
        self.b_p = np.repeat(-1.0, self.num_agents)
        self.a_s = np.repeat(0.0, self.num_agents)
        self.b_s = np.repeat(-1.0, self.num_agents)
        self.opt_sog = self.length / 1000.
        self.school_sog = self.length / 1000.
        self.ucrit = self.length / 1000. * 1.6
        self.is_stuck = np.zeros(self.num_agents, dtype=bool)
        self.agents_within_buffers = [np.array([], dtype=int) for _ in range(self.num_agents)]
        self.nearest_neighbor_distance = np.full(self.num_agents, np.nan)
        self.closest_agent = np.full(self.num_agents, np.nan)
        self.in_eddy = np.zeros(self.num_agents, dtype=bool)
        self.time_since_eddy_escape = np.zeros(self.num_agents, dtype=float)
        self.max_eddy_escape_seconds = 1000

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

        # create standard datasets using io helper to keep logic centralized
        sim_state = {
            'num_agents': self.num_agents,
            'num_timesteps': self.num_timesteps,
            'sex': self.sex,
            'length': self.length,
            'weight': self.weight,
            'body_depth': self.body_depth,
            'metadata': {'model_name': self.model_name}
        }
        try:
            io.write_sim_initial(self.db, sim_state)
        except Exception:
            # fallback to manual creation if write_sim_initial fails
            hdf5_io.write_dataset(self.db, "agent_data/sex", np.zeros((self.num_agents,), dtype=np.int8))
            hdf5_io.write_dataset(self.db, "agent_data/length", np.zeros((self.num_agents,), dtype=np.float32))
            hdf5_io.write_dataset(self.db, "agent_data/weight", np.zeros((self.num_agents,), dtype=np.float32))
            hdf5_io.write_dataset(self.db, "agent_data/body_depth", np.zeros((self.num_agents,), dtype=np.float32))

        # populate agent attributes using the agents module
        agents.sim_sex(self)
        agents.sim_length(self, fish_length)
        agents.sim_weight(self)
        agents.sim_body_depth(self)

        # write agent attributes into HDF5 static datasets
        hdf5_io.write_dataset(self.db, "agent_data/sex", self.sex)
        hdf5_io.write_dataset(self.db, "agent_data/length", self.length)
        hdf5_io.write_dataset(self.db, "agent_data/weight", self.weight)
        hdf5_io.write_dataset(self.db, "agent_data/body_depth", self.body_depth)
        # also write legacy top-level datasets for compatibility
        hdf5_io.write_dataset(self.db, "sex", self.sex)
        hdf5_io.write_dataset(self.db, "length", self.length)
        hdf5_io.write_dataset(self.db, "weight", self.weight)
        hdf5_io.write_dataset(self.db, "body_depth", self.body_depth)
        if hasattr(self.db, 'flush'):
            try:
                self.db.flush()
            except Exception:
                pass

        # import any provided environment files (fail loudly during development)
        for ef in self.env_files:
            _ = io.enviro_import(ef)

        # ensure minimal environment placeholders exist so downstream modules
        # that read environment/* will have something to sample in unit tests
        hdf5_io.create_environment_placeholders(self.db)

        # small helpers: construct movement and behavior helpers now
        self._movement = movement_mod.movement(self)
        self._behavior = behavior_mod.behavior(1.0, self)
        self._fatigue = None

    def timestep(self, t, dt, g=None, pid_controller=None):
        # Advance time and run a single simulation timestep integrating
        # behavior -> fatigue -> movement -> write outputs.
        self.cumulative_time += dt

        # keep previous positions for velocity calculations
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()

        # ensure a pid controller is available
        pid = pid_controller or self.pid_controller

        # mask of agents able to move
        mask = np.where(self.dead == 0, True, False)

        # instantiate per-timestep helpers
        try:
            behavior = behavior_mod.behavior(dt, self)
        except Exception:
            behavior = self._behavior

        try:
            fatigue = fatigue_mod.fatigue(t, dt, self)
        except Exception:
            fatigue = None

        try:
            movement = movement_mod.movement(self)
        except Exception:
            movement = self._movement

        # run fatigue assessment first to update battery / swim modes
        if fatigue is not None:
            try:
                fatigue.assess_fatigue()
            except Exception:
                pass

        # behavior arbitration produces desired heading vector
        try:
            new_heading = behavior.arbitrate(t)
            # behavior.arbitrate may return scalar or array
            self.heading = np.array(new_heading, dtype=np.float32)
        except Exception:
            # keep existing heading
            pass

        # calculate movement-related quantities
        try:
            if movement is not None:
                movement.frequency(mask, t, dt)
                movement.thrust_fun(mask, t, dt)
                movement.drag_fun(mask, t, dt)
                dxdy = movement.swim(t, dt, pid or pid_controller, mask)
            else:
                dxdy = np.zeros((self.num_agents, 2), dtype=np.float32)
        except Exception:
            dxdy = np.zeros((self.num_agents, 2), dtype=np.float32)

        # apply movement
        try:
            self.X = self.X + dxdy[:, 0]
            self.Y = self.Y + dxdy[:, 1]
        except Exception:
            # fallback scalar handling
            self.X = self.X + dxdy
            self.Y = self.Y + dxdy

        # update velocities
        try:
            self.x_vel = (self.X - self.prev_X) / dt
            self.y_vel = (self.Y - self.prev_Y) / dt
        except Exception:
            pass

        # write minimal outputs back to HDF5 for downstream consumers
        hdf5_io.write_dataset(self.db, 'X', self.X)
        hdf5_io.write_dataset(self.db, 'Y', self.Y)
        hdf5_io.write_dataset(self.db, 'prev_X', self.prev_X)
        hdf5_io.write_dataset(self.db, 'prev_Y', self.prev_Y)

        # write per-timestep slices into time-indexed agent_data arrays
        h5 = hdf5_io.get_hdf5_obj(self)
        ts = int(max(0, min(int(self.cumulative_time) - 1, self.num_timesteps - 1)))
        tracked = ('agent_data/X', 'agent_data/Y', 'agent_data/prev_X', 'agent_data/prev_Y', 'agent_data/ideal_sog', 'agent_data/Hz')
        for key in tracked:
            arr = hdf5_io.read_dataset(h5, key, default=None)
            if arr is None:
                logging.getLogger(__name__).debug('Dataset %s missing; skipping timestep write', key)
                continue
            attr_key = key.split('/')[-1]
            # try direct attribute, then lowercase, then capitalized
            if hasattr(self, attr_key):
                val = getattr(self, attr_key)
            elif hasattr(self, attr_key.lower()):
                val = getattr(self, attr_key.lower())
            elif hasattr(self, attr_key.capitalize()):
                val = getattr(self, attr_key.capitalize())
            else:
                logging.getLogger(__name__).debug('No matching attribute for %s; skipping', attr_key)
                continue
            arr[:, ts] = np.array(val)
            hdf5_io.write_dataset(h5, key, arr)

        # flush when supported (best-effort)
        if hasattr(self.db, 'flush'):
            try:
                self.db.flush()
            except Exception:
                pass

        return True

    def run(self, model_name=None, n=1, dt=1.0, video=False, k_p=None, k_i=None, k_d=None):
        # Simple run loop that calls `timestep` n times
        # allow caller supplied PID tuning values
        if self.pid_controller is not None and k_p is not None:
            try:
                self.pid_controller.k_p = np.array([k_p])
                self.pid_controller.k_i = np.array([k_i]) if k_i is not None else self.pid_controller.k_i
                self.pid_controller.k_d = np.array([k_d]) if k_d is not None else self.pid_controller.k_d
            except Exception:
                pass

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
