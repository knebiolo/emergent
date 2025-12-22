import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import CubicSpline

from .io_utils import initialize_hdf5, enviro_import
from .pid import PID_controller


class simulation:
    """Core simulation class extracted from the monolith.

    This is a first-pass extraction that keeps the original `__init__`
    behavior but delegates HDF initialization and env imports to `io_utils`.
    Further refactoring will split large methods into focused helpers.
    """

    def __init__(self,
                 model_dir,
                 model_name,
                 crs,
                 basin,
                 water_temp,
                 start_polygon,
                 centerline,
                 env_files=None,
                 fish_length=None,
                 num_timesteps=100,
                 num_agents=100,
                 use_gpu=False,
                 pid_tuning=False,
                 hecras_plan_path=None,
                 hecras_fields=None,
                 hecras_k=8,
                 use_hecras=False,
                 hecras_write_rasters=False,
                 defer_hdf=False,
                 defer_log_dir=None,
                 defer_log_fmt='npz'):

        self.arr = np  # placeholder for get_arr(use_gpu)
        # HDF/logging config
        self.defer_hdf = defer_hdf
        self.defer_log_dir = defer_log_dir
        self.defer_log_fmt = defer_log_fmt

        # basic params
        self.model_dir = model_dir
        self.model_name = model_name
        self.start_polygon_path = start_polygon
        self.db = os.path.join(self.model_dir, f"{self.model_name}.h5")
        self.crs = crs
        self.basin = basin
        self.num_agents = num_agents
        self.num_timesteps = num_timesteps
        self.water_temp = water_temp

        # PID attach
        try:
            self.pid_controller = PID_controller(self.num_agents, k_p=0.5, k_i=0.0, k_d=0.1)
            try:
                self.pid_controller.interp_PID()
            except Exception:
                pass
        except Exception:
            self.pid_controller = None

        # initialize agent properties
        self.sim_sex()
        self.sim_length(fish_length)
        self.sim_weight()
        self.sim_body_depth()

        # buffers
        self.flush_interval = 50
        self._hdf5_buffers = {}
        self._buffer_pos = 0

        # Create HDF file and initialize datasets
        try:
            import h5py
            self.hdf5 = h5py.File(self.db, 'w')
            initialize_hdf5(self)
        except Exception:
            self.hdf5 = None

        # Preload env files if provided
        if isinstance(env_files, dict):
            for key, fp in env_files.items():
                if not fp:
                    continue
                path = fp if os.path.isabs(fp) else os.path.join(self.model_dir, fp)
                if os.path.exists(path):
                    try:
                        enviro_import(self, path, key)
                    except Exception:
                        pass

    # ---- minimal implementations of helper methods used in __init__ ----
    def sim_sex(self):
        self.sex = np.random.choice([0, 1], size=self.num_agents)

    def sim_length(self, fish_length=None):
        if fish_length is not None:
            self.length = np.repeat(fish_length, self.num_agents)
        else:
            self.length = np.random.lognormal(mean=6.39, sigma=0.07, size=self.num_agents)
        self.length = np.where(self.length < 475., 475., self.length)
        self.sog = self.length / 1000.0
        self.ideal_sog = self.sog
        self.ucrit = self.sog * 1.6

    def sim_weight(self):
        self.weight = (0.0155 * (self.length / 10.0) ** 3) / 1000.0

    def sim_body_depth(self):
        self.body_depth = np.exp(-1.938 + np.log(self.length) * 1.084) / 10.0
        self.too_shallow = self.body_depth / 100.0 / 2.0
        self.opt_wat_depth = self.body_depth / 100.0 * 3.0 + self.too_shallow
