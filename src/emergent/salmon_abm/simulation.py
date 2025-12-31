"""Skeleton `simulation` class extracted from sockeye.py.

This minimal class preserves the original constructor signature and
provides `timestep`, `run`, and `close` methods so other code can import
and be migrated incrementally. The implementation is intentionally light
weight to remain testable without heavy environment data.
"""
import os
import tempfile
import sys
import h5py
import numpy as np
import logging
from scipy.ndimage import distance_transform_edt
try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None
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
        # Output write cadence (timesteps). Defaults preserve existing behavior.
        # - write_frequency <= 0: skip writing agent_data/* time-series each step.
        # - flush_frequency <= 0: never flush each step (caller can flush/close).
        self.write_frequency = int(getattr(self, 'write_frequency', 1) or 0)
        # Leave `flush_frequency` unset by default so it can follow `write_frequency`
        # when callers change cadence after initialization (e.g. profiling harness).
        self.flush_frequency = getattr(self, 'flush_frequency', None)
        # Cache for h5py dataset handles used in per-step writes
        self._timeseries_ds_cache = {}
        # Cache for static dataset arrays (e.g., environment rasters) to avoid
        # repeatedly reading large HDF5 datasets each timestep.
        self._dataset_cache = {}
        # When True, skip all per-step writes to the HDF5/db store (X/Y and
        # agent_data/*). This is useful for "watch-only" runs where you want
        # realtime behavior without I/O overhead.
        try:
            self.disable_hdf_writes = bool(getattr(self, 'disable_hdf_writes', False))
        except Exception:
            self.disable_hdf_writes = False
        # Back-compat alias (some callers may prefer this name)
        try:
            self.disable_output_writes = bool(getattr(self, 'disable_output_writes', self.disable_hdf_writes))
        except Exception:
            self.disable_output_writes = self.disable_hdf_writes

        # Neighbor-finding configuration.
        # - Default sensing radius is a fixed 1 meter (simple + predictable).
        #   To use body-length scaling instead, set `neighbor_buffer_radius <= 0`
        #   and configure `neighbor_buffer_lengths` (default: 2.0).
        # - Neighbor graph rebuild is throttled via `neighbor_update_seconds` because
        #   neighbors do not change meaningfully every sub-second timestep.
        self.neighbor_update_seconds = float(getattr(self, 'neighbor_update_seconds', 2.0))
        self.neighbor_update_interval_steps = int(getattr(self, 'neighbor_update_interval_steps', 0) or 0)
        self.neighbor_buffer_lengths = float(getattr(self, 'neighbor_buffer_lengths', 2.0))
        # Explicit override in meters; default is 1.0m. Set <=0 to fall back to
        # body-length scaling via `neighbor_buffer_lengths`.
        try:
            self.neighbor_buffer_radius = float(getattr(self, 'neighbor_buffer_radius', 1.0))
        except Exception:
            self.neighbor_buffer_radius = 1.0
        self._neighbor_last_build_step = None

        # Per-timestep cache for sampled environment values (dedupe repeated calls
        # to `sample_environment` across multiple cues).
        self.cache_env_samples = bool(getattr(self, 'cache_env_samples', True))
        self._env_sample_cache = {}
        self._env_sample_cache_step = None
        self._env_sample_cache_gen = 0
        # Per-timestep cache for computed pixel indices from `geo_to_pixel` keyed
        # by transform (independent of raster_name). This avoids repeating the
        # coordinate transform when sampling multiple rasters with the same grid.
        self._env_pixel_cache = {}

        # Avoid/mental-map configuration. In non-debug runs, prefer a sparse,
        # per-agent history representation to avoid per-agent HDF5 raster costs.
        self.use_sparse_avoid_memory = bool(getattr(self, 'use_sparse_avoid_memory', True))
        self.avoid_history_len = int(getattr(self, 'avoid_history_len', 1024))
        # Cap the number of history entries considered when computing the avoid
        # ("already_been_here") repulsion force. Set <=0 to consider all entries.
        self.avoid_force_history_len = int(getattr(self, 'avoid_force_history_len', 128))
        self.avoid_memory_horizon_s = float(getattr(self, 'avoid_memory_horizon_s', 7200.0))
        # Only persist dense per-agent memory rasters when explicitly requested.
        self.persist_avoid_memory_hdf5 = bool(getattr(self, 'persist_avoid_memory_hdf5', False))
        # Internal sparse history storage (allocated lazily)
        self.avoid_hist_rows = None
        self.avoid_hist_cols = None
        self.avoid_hist_t = None
        self.avoid_hist_pos = None
        self._avoid_map_shape = None

        # prepare simple RNG to keep behavior deterministic when seed used
        try:
            # agents may set RNG via self.rng if needed
            self.rng = np.random.default_rng()
        except Exception:
            self.rng = None
        # compatibility: some legacy code expects `sim.arr.random.choice`
        # simplest approach: expose the numpy module under `self.arr`
        try:
            self.arr = np
        except Exception:
            self.arr = None
        # always create a PID controller instance (safe defaults)
        # PID tuning can still be enabled via `pid_tuning` flag
        try:
            self.pid_controller = pid.PID_controller(self.num_agents)
        except Exception:
            self.pid_controller = None
        
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
        # fish velocity over ground (distinct from water velocity rasters)
        self.fish_x_vel = np.zeros(self.num_agents, dtype=np.float32)
        self.fish_y_vel = np.zeros(self.num_agents, dtype=np.float32)
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
        # When agents are fatigued (swim_behav == 3), their effective sustainable
        # swim speed may be reduced. Keep this as a separate array so callers can
        # tune fatigued capacity without mutating the baseline `max_s_U`.
        #
        # Units: body-lengths / second (BL/s), matching legacy usage.
        self.fatigued_max_s_U_multiplier = float(getattr(self, 'fatigued_max_s_U_multiplier', 1.0))
        try:
            base = np.asarray(self.max_s_U, dtype=np.float32)
        except Exception:
            base = np.repeat(2.77, self.num_agents).astype(np.float32)
        self.max_s_U_fatigued = (base * self.fatigued_max_s_U_multiplier).astype(np.float32)
        self.a_p = np.repeat(0.0, self.num_agents)
        self.b_p = np.repeat(-1.0, self.num_agents)
        self.a_s = np.repeat(0.0, self.num_agents)
        self.b_s = np.repeat(-1.0, self.num_agents)
        # derived speeds are initialized after agents.sim_length() populates `self.length`
        self.opt_sog = np.zeros(self.num_agents, dtype=np.float32)
        self.school_sog = np.zeros(self.num_agents, dtype=np.float32)
        self.ucrit = np.zeros(self.num_agents, dtype=np.float32)
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

        # Derived quantities that depend on agent attributes (length/body_depth, etc.).
        # These must be computed after the agents module populates the base attributes.
        try:
            self.opt_sog = (self.length / 1000.0).astype(np.float32)
            self.school_sog = (self.length / 1000.0).astype(np.float32)
            self.ucrit = (self.length / 1000.0 * 1.6).astype(np.float32)
            # initialize ideal_sog and sog to a non-zero default when unset
            if not np.any(np.asarray(self.ideal_sog)):
                self.ideal_sog = self.school_sog.copy()
            if not np.any(np.asarray(self.sog)):
                self.sog = self.ideal_sog.copy()
        except Exception:
            pass

        # Refugia definition (canonical): places where a fatigued fish can hold
        # station (i.e., water velocity does not exceed sustainable fatigued
        # capacity). The environment layer is computed relative to a reference
        # fish length (mm) to avoid per-agent maps.
        try:
            ref_len = getattr(self, 'refugia_ref_length_mm', None)
            if ref_len is None:
                ref_len = float(np.nanmedian(np.asarray(self.length, dtype=float)))
            if not np.isfinite(ref_len) or ref_len <= 0:
                ref_len = 500.0
            self.refugia_ref_length_mm = float(ref_len)
        except Exception:
            self.refugia_ref_length_mm = 500.0
        # Derive `environment/refugia` automatically when possible.
        try:
            self.auto_derive_refugia = bool(getattr(self, 'auto_derive_refugia', True))
        except Exception:
            self.auto_derive_refugia = True
        self._refugia_derived = False
        # Refugia cue sensing/search radius (meters). Default is a fixed 1m to
        # avoid per-agent radii and keep behavior predictable. Set <=0 to
        # disable gating (always point toward nearest refugia cell).
        try:
            self.refugia_search_radius_m = float(getattr(self, 'refugia_search_radius_m', 1.0))
        except Exception:
            self.refugia_search_radius_m = 1.0

        # If a start polygon was provided, sample initial agent positions inside it
        if start_polygon:
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                gdf = gpd.read_file(start_polygon)
                if gdf is None or len(gdf) == 0:
                    raise RuntimeError('start polygon shapefile empty')
                geom = gdf.unary_union if len(gdf) > 1 else gdf.geometry.iloc[0]
                minx, miny, maxx, maxy = geom.bounds
                rng = getattr(self, 'rng', None)
                if rng is None:
                    rng = np.random.default_rng()
                pts = []
                attempts = 0
                # draw random points within bbox and test containment
                while len(pts) < self.num_agents and attempts < max(5000, self.num_agents * 100):
                    x = float(rng.uniform(minx, maxx))
                    y = float(rng.uniform(miny, maxy))
                    if geom.contains(Point(x, y)):
                        pts.append((x, y))
                    attempts += 1
                # fallback: use representative point / centroid if sampling failed
                if len(pts) < self.num_agents:
                    rep = geom.representative_point()
                    rx, ry = float(rep.x), float(rep.y)
                    while len(pts) < self.num_agents:
                        pts.append((rx, ry))
                xs = np.array([p[0] for p in pts], dtype=np.float64)
                ys = np.array([p[1] for p in pts], dtype=np.float64)
                # assign to simulation state
                self.X = xs
                self.Y = ys
                self.prev_X = xs.copy()
                self.prev_Y = ys.copy()
                # write initial positions into HDF5 (top-level and time-indexed arrays)
                try:
                    hdf5_io.write_dataset(self.db, 'X', self.X)
                    hdf5_io.write_dataset(self.db, 'Y', self.Y)
                    hdf5_io.write_timeseries_step(self.db, 'agent_data/X', 0, self.X)
                    hdf5_io.write_timeseries_step(self.db, 'agent_data/Y', 0, self.Y)
                except Exception:
                    pass
                # record that start polygon was used
                try:
                    hdf5_io.write_dataset(self.db, 'metadata/start_polygon', os.path.basename(start_polygon))
                except Exception:
                    pass
            except Exception:
                # do not fail initialization for missing or invalid polygon
                pass

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
        # Import provided environment rasters into the simulation HDF5 DB and
        # set raster transform attributes using the centralized helper.
        try:
            for ef in self.env_files:
                try:
                    base = os.path.splitext(os.path.basename(ef))[0]
                    # helper writes into HDF5 and returns the transform tuple
                    arr, tr_tup, crs = io.write_raster_to_hdf5(self.db, ef, dataset_name=base, sim=self)
                except Exception:
                    # best-effort per-file: continue on error
                    continue
        except Exception:
            # non-fatal: proceed even if env file handling fails
            pass

        # ensure minimal environment placeholders exist so downstream modules
        # that read environment/* will have something to sample in unit tests
        hdf5_io.create_environment_placeholders(self.db)

        # best-effort: compute `environment/distance_to` when missing and depth exists
        try:
            h5 = hdf5_io.get_hdf5_obj(self)
            dist_ds = hdf5_io.read_dataset(h5, 'environment/distance_to', default=None)
            if dist_ds is None:
                depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=None)
                if depth_ds is not None:
                    depth_arr = np.asarray(depth_ds, dtype=float)
                    if depth_arr.ndim == 2 and depth_arr.size > 1:
                        wetted = np.isfinite(depth_arr) & (depth_arr != -9999.0)
                        try:
                            tr = getattr(self, 'depth_rast_transform', None)
                            pw = float(tr[0]) if tr is not None else 1.0
                        except Exception:
                            pw = 1.0
                        dist_to_bound = distance_transform_edt(wetted) * abs(pw)
                        hdf5_io.write_dataset(h5, 'environment/distance_to', dist_to_bound.astype('float32'))
        except Exception:
            pass

        # Movement-related defaults required by movement helpers. Set early so
        # movement.frequency/drag_fun/swim can run safely even if attributes
        # are not later mutated.
        try:
            self.pid_tuning = pid_tuning
        except Exception:
            self.pid_tuning = False
        try:
            self.wave_drag = np.ones(self.num_agents, dtype=float)
        except Exception:
            self.wave_drag = np.ones(self.num_agents)
        try:
            self.Hz = np.zeros(self.num_agents, dtype=float)
        except Exception:
            self.Hz = np.zeros(self.num_agents)
        # provide a drag_coeff callable expected by movement; will be overridden
        # later if movement helper is available.
        try:
            self.drag_coeff = lambda reynolds: np.interp(reynolds, [2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5], [0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])
        except Exception:
            self.drag_coeff = lambda reynolds: np.ones_like(reynolds) * 0.12

        # Ensure raster transforms and coordinate grids exist without clobbering
        # real-world transforms written by `io.write_raster_to_hdf5`.
        try:
            h5 = hdf5_io.get_hdf5_obj(self)
            depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=None)
            if depth_ds is not None:
                depth_arr = np.asarray(depth_ds)
                if depth_arr.ndim == 2 and depth_arr.size > 1:
                    nrows, ncols = depth_arr.shape
                    # use existing depth raster transform when available; otherwise fall back
                    transform = getattr(self, 'depth_rast_transform', None)
                    if transform is None:
                        transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                        self.depth_rast_transform = transform
                    # ensure related transforms exist
                    for attr in ('vel_mag_rast_transform', 'vel_dir_rast_transform', 'vel_x_rast_transform', 'vel_y_rast_transform', 'refugia_map_transform'):
                        if getattr(self, attr, None) is None:
                            try:
                                setattr(self, attr, transform)
                            except Exception:
                                pass

                    # Only write coordinate grids when missing or mismatched shape.
                    try:
                        existing_x = hdf5_io.read_dataset(h5, 'environment/x_coords', default=None)
                        existing_y = hdf5_io.read_dataset(h5, 'environment/y_coords', default=None)
                        have_ok = False
                        try:
                            if existing_x is not None and existing_y is not None:
                                have_ok = (np.asarray(existing_x).shape == (nrows, ncols)) and (np.asarray(existing_y).shape == (nrows, ncols))
                        except Exception:
                            have_ok = False
                        if not have_ok:
                            # compute from affine transform (supports Affine-like objects or 6-tuples)
                            try:
                                a = float(getattr(transform, 'a', transform[0]))
                                b = float(getattr(transform, 'b', transform[1]))
                                c = float(getattr(transform, 'c', transform[2]))
                                d = float(getattr(transform, 'd', transform[3]))
                                e = float(getattr(transform, 'e', transform[4]))
                                f = float(getattr(transform, 'f', transform[5]))
                                cols = np.arange(ncols, dtype=float)
                                rows = np.arange(nrows, dtype=float)
                                col_indices, row_indices = np.meshgrid(cols, rows)
                                x_coords = a * col_indices + b * row_indices + c
                                y_coords = d * col_indices + e * row_indices + f
                                hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
                                hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            # tolerate any failures here; behavior will be more limited but simulation can still run
            if getattr(self, 'depth_rast_transform', None) is None:
                self.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # Best-effort: derive an environment refugia mask using the canonical
        # fatigued station-holding definition.
        try:
            if getattr(self, 'auto_derive_refugia', False):
                self.derive_environment_refugia()
        except Exception:
            pass

        # heading initialization deferred until behavior helper is available

        # small helpers: construct movement and behavior helpers now
        self._movement = movement_mod.movement(self)
        self._behavior = behavior_mod.behavior(1.0, self)
        # If debug flags are enabled, start the diagnostics worker to accept queued writes
        try:
            if getattr(self, 'debug_behavior', False) or getattr(self, 'debug_movement', False):
                try:
                    if hasattr(self._behavior, '_start_diag_thread'):
                        self._behavior._start_diag_thread()
                except Exception:
                    pass
        except Exception:
            pass
        # Initialize headings by sampling rasters from the DB (callable so
        # external code can re-run initialization after injecting rasters).
        import logging
        try:
            self.initialize_headings_from_db()
        except Exception as e:
            logging.getLogger(__name__).warning('initialize_headings_from_db failed during simulation init: %s', e)
        # set initial fish velocity so agents start with non-zero fish velocity
        try:
            fv_x = self.ideal_sog * np.cos(self.heading)
            fv_y = self.ideal_sog * np.sin(self.heading)
            self.initial_fish_vel = np.stack((fv_x, fv_y), axis=1)
        except Exception:
            self.initial_fish_vel = np.zeros((self.num_agents, 2), dtype=float)
        self._fatigue = None

    def _neighbor_radius_m(self) -> float:
        """Return the neighbor buffer radius in meters.

        Uses `neighbor_buffer_radius` when set and >0 (meters).

        If `neighbor_buffer_radius <= 0`, computes:
            `neighbor_buffer_lengths * median(length_m)`
        where `length` is in mm.
        """
        # fixed-radius override in meters
        try:
            r = getattr(self, 'neighbor_buffer_radius', None)
            if r is not None:
                r = float(r)
                if np.isfinite(r) and r > 0.0:
                    return float(r)
        except Exception:
            pass

        try:
            bl = float(getattr(self, 'neighbor_buffer_lengths', 2.0))
        except Exception:
            bl = 2.0
        if not np.isfinite(bl) or bl <= 0.0:
            bl = 2.0

        try:
            length_mm = np.asarray(getattr(self, 'length', np.array([500.0])), dtype=float)
            length_m = np.nanmedian(length_mm) / 1000.0
        except Exception:
            length_m = 0.5
        if not np.isfinite(length_m) or length_m <= 0.0:
            length_m = 0.5
        return float(bl * length_m)

    def _env_sample_cache_key(self, transform, raster_name: str):
        try:
            rn = str(raster_name)
        except Exception:
            rn = raster_name
        try:
            gen = int(getattr(self, '_env_sample_cache_gen', 0) or 0)
        except Exception:
            gen = 0

        tr_key = None
        if transform is not None:
            try:
                vals = tuple(transform)
                if len(vals) >= 6:
                    tr_key = tuple(float(vals[i]) for i in range(6))
                else:
                    tr_key = tuple(float(v) for v in vals)
            except Exception:
                tr_key = id(transform)

        return (rn, tr_key, gen)

    def _env_pixel_cache_key(self, transform):
        try:
            gen = int(getattr(self, '_env_sample_cache_gen', 0) or 0)
        except Exception:
            gen = 0

        tr_key = None
        if transform is not None:
            try:
                vals = tuple(transform)
                if len(vals) >= 6:
                    tr_key = tuple(float(vals[i]) for i in range(6))
                else:
                    tr_key = tuple(float(v) for v in vals)
            except Exception:
                tr_key = id(transform)

        return (tr_key, gen)

    def initialize_headings_from_db(self):
        """(Re)initialize `self.heading` by sampling velocity rasters in the sim DB.

        This is exposed as a method so callers can write rasters into `sim.db`
        after construction and then re-run initialization.
        """
        h5 = hdf5_io.get_hdf5_obj(self)
        if h5 is None:
            return False
        heading_set = False
        # try raw component rasters first
        vel_x_ds = hdf5_io.read_dataset(h5, 'environment/vel_x', default=None)
        vel_y_ds = hdf5_io.read_dataset(h5, 'environment/vel_y', default=None)
        try:
            if vel_x_ds is not None and vel_y_ds is not None:
                vel_x_arr = np.array(vel_x_ds)
                vel_y_arr = np.array(vel_y_ds)
                from emergent.salmon_abm.utils import geo_to_pixel
                try:
                    rows, cols = geo_to_pixel(self.X, self.Y, self.depth_rast_transform)
                    rows = np.asarray(rows, dtype=int)
                    cols = np.asarray(cols, dtype=int)
                except Exception:
                    # fallback: use x_coords/y_coords arrays stored in the DB to find nearest pixels
                    x_coords_ds = hdf5_io.read_dataset(h5, 'environment/x_coords', default=None)
                    y_coords_ds = hdf5_io.read_dataset(h5, 'environment/y_coords', default=None)
                    if x_coords_ds is None or y_coords_ds is None:
                        raise
                    x_coords = np.asarray(x_coords_ds)
                    y_coords = np.asarray(y_coords_ds)
                    # x_coords assumed shape (nrows, ncols): pick nearest column per-agent
                    # compute difference between agent X and each column (broadcast)
                    # handle shapes carefully
                    try:
                        # nearest column: minimize |x_coords[row0, col] - X|
                        cols = np.argmin(np.abs(x_coords[0:1, :] - self.X[:, None]), axis=1)
                        # nearest row: minimize |y_coords[row, col0] - Y|
                        rows = np.argmin(np.abs(y_coords[:, 0:1] - self.Y[None, :]), axis=0)
                        rows = np.asarray(rows, dtype=int)
                        cols = np.asarray(cols, dtype=int)
                    except Exception:
                        # last-resort: zeros
                        rows = np.zeros(self.num_agents, dtype=int)
                        cols = np.zeros(self.num_agents, dtype=int)
                valid = (rows >= 0) & (cols >= 0) & (rows < vel_x_arr.shape[0]) & (cols < vel_x_arr.shape[1])
                vx = np.full(self.num_agents, np.nan)
                vy = np.full(self.num_agents, np.nan)
                if np.any(valid):
                    vx[valid] = vel_x_arr[rows[valid], cols[valid]]
                    vy[valid] = vel_y_arr[rows[valid], cols[valid]]
                # Record sampled water velocity components on simulation so alignment
                # and other cues can use neighbor velocities before movement updates.
                try:
                    self.x_vel = np.where(np.isnan(vx), 0.0, vx).astype(np.float32)
                    self.y_vel = np.where(np.isnan(vy), 0.0, vy).astype(np.float32)
                except Exception:
                    pass
                both_nan = np.isnan(vx) & np.isnan(vy)
                raw_heading = np.arctan2(-vy, -vx)
                raw_heading = np.where(both_nan, self.heading, raw_heading)
                self.heading = np.asarray(raw_heading, dtype=np.float32)
                heading_set = True
        except Exception:
            heading_set = False

        # fallback to magnitude+direction rasters
        if not heading_set:
            vel_mag_ds = hdf5_io.read_dataset(h5, 'environment/vel_mag', default=None)
            vel_dir_ds = hdf5_io.read_dataset(h5, 'environment/vel_dir', default=None)
            try:
                if vel_mag_ds is not None and vel_dir_ds is not None:
                    mag = np.array(vel_mag_ds)
                    vdir = np.array(vel_dir_ds)
                    from emergent.salmon_abm.utils import geo_to_pixel
                    rows, cols = geo_to_pixel(self.X, self.Y, self.depth_rast_transform)
                    rows = np.asarray(rows, dtype=int)
                    cols = np.asarray(cols, dtype=int)
                    valid = (rows >= 0) & (cols >= 0) & (rows < mag.shape[0]) & (cols < mag.shape[1])
                    vx = np.full(self.num_agents, np.nan)
                    vy = np.full(self.num_agents, np.nan)
                    if np.any(valid):
                        vals_mag = mag[rows[valid], cols[valid]]
                        vals_dir = vdir[rows[valid], cols[valid]]
                        vx[valid] = vals_mag * np.cos(vals_dir)
                        vy[valid] = vals_mag * np.sin(vals_dir)
                    try:
                        self.x_vel = np.where(np.isnan(vx), 0.0, vx).astype(np.float32)
                        self.y_vel = np.where(np.isnan(vy), 0.0, vy).astype(np.float32)
                    except Exception:
                        pass
                    both_nan = np.isnan(vx) & np.isnan(vy)
                    raw_heading = np.arctan2(-vy, -vx)
                    raw_heading = np.where(both_nan, self.heading, raw_heading)
                    self.heading = np.asarray(raw_heading, dtype=np.float32)
                    heading_set = True
            except Exception:
                heading_set = False

        # Persist initial ideal_sog into the HDF5 time-indexed array (column 0)
        try:
            h5 = hdf5_io.get_hdf5_obj(self)
            arr = hdf5_io.read_dataset(h5, 'agent_data/ideal_sog', default=None)
            if arr is not None:
                try:
                    arr[:, 0] = np.array(self.ideal_sog)
                    hdf5_io.write_dataset(h5, 'agent_data/ideal_sog', arr)
                except Exception:
                    pass
        except Exception:
            pass

        return heading_set

    def initialize_mental_map(self, avoid_cell_size: float | None = None, *, create_datasets: bool = True) -> bool:
        """Create per-agent memory rasters and `mental_map_transform` for the avoid cue.

        This is a lightweight analogue of sockeye.py's mental map initialization,
        sized from the depth raster extent and a configurable coarse cell size.
        """
        h5 = hdf5_io.get_hdf5_obj(self)
        if h5 is None:
            return False
        if avoid_cell_size is None:
            avoid_cell_size = float(getattr(self, 'avoid_cell_size', 10.0))
        if avoid_cell_size <= 0:
            avoid_cell_size = 10.0

        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=None)
        if depth_ds is None:
            return False
        depth_arr = np.asarray(depth_ds)
        if depth_arr.ndim != 2 or depth_arr.size <= 1:
            return False
        nrows, ncols = depth_arr.shape

        # infer depth pixel size and origin from the raster transform
        tr = getattr(self, 'depth_rast_transform', None) or (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        try:
            a = float(getattr(tr, 'a', tr[0]))
            b = float(getattr(tr, 'b', tr[1]))
            c = float(getattr(tr, 'c', tr[2]))
            d = float(getattr(tr, 'd', tr[3]))
            e = float(getattr(tr, 'e', tr[4]))
            f = float(getattr(tr, 'f', tr[5]))
        except Exception:
            a, b, c, d, e, f = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        pw = abs(a) if a != 0 else 1.0
        ph = abs(e) if e != 0 else 1.0
        width_m = ncols * pw
        height_m = nrows * ph
        avoid_width = int(np.ceil(width_m / avoid_cell_size)) + 1
        avoid_height = int(np.ceil(height_m / avoid_cell_size)) + 1
        avoid_width = max(1, avoid_width)
        avoid_height = max(1, avoid_height)

        # north-up coarse grid transform anchored to the depth raster origin
        self.mental_map_transform = (avoid_cell_size, 0.0, c, 0.0, -avoid_cell_size, f)
        self._avoid_map_shape = (avoid_height, avoid_width)

        if create_datasets:
            # ensure memory datasets exist for each agent (create lazily if missing)
            for i in range(int(self.num_agents)):
                key = f'memory/{i}'
                existing = None
                if key in h5:
                    try:
                        existing = h5[key]
                    except Exception:
                        existing = hdf5_io.read_dataset(h5, key, default=None)
                if existing is not None:
                    try:
                        if np.asarray(existing).shape == (avoid_height, avoid_width):
                            continue
                    except Exception:
                        pass
                hdf5_io.write_dataset(h5, key, np.full((avoid_height, avoid_width), np.nan, dtype=np.float32))
        return True

    def ensure_avoid_history(self, history_len: int | None = None) -> bool:
        n = int(getattr(self, 'num_agents', 0) or 0)
        if n <= 0:
            return False
        if history_len is None:
            history_len = int(getattr(self, 'avoid_history_len', 1024))
        history_len = max(1, int(history_len))
        # allocate or resize (best-effort)
        try:
            rows = getattr(self, 'avoid_hist_rows', None)
            cols = getattr(self, 'avoid_hist_cols', None)
            ts = getattr(self, 'avoid_hist_t', None)
            pos = getattr(self, 'avoid_hist_pos', None)
            if rows is not None and cols is not None and ts is not None and pos is not None:
                if np.asarray(rows).shape == (n, history_len):
                    return True
        except Exception:
            pass

        try:
            self.avoid_hist_rows = np.full((n, history_len), -1, dtype=np.int16)
            self.avoid_hist_cols = np.full((n, history_len), -1, dtype=np.int16)
            self.avoid_hist_t = np.full((n, history_len), np.nan, dtype=np.float32)
            self.avoid_hist_pos = np.zeros((n,), dtype=np.int32)
            return True
        except Exception:
            return False

    def seed_avoid_history(self, rows: np.ndarray, cols: np.ndarray, t: float) -> bool:
        """Seed sparse avoid history at the current write position for each agent."""
        if not self.ensure_avoid_history():
            return False
        rows = np.asarray(rows, dtype=int).reshape((-1,))
        cols = np.asarray(cols, dtype=int).reshape((-1,))
        n = int(self.num_agents)
        if rows.size != n or cols.size != n:
            return False
        idx = np.arange(n, dtype=int)
        pos = np.asarray(self.avoid_hist_pos, dtype=int)
        self.avoid_hist_rows[idx, pos] = rows.astype(np.int16)
        self.avoid_hist_cols[idx, pos] = cols.astype(np.int16)
        self.avoid_hist_t[idx, pos] = float(t)
        self.avoid_hist_pos = ((pos + 1) % self.avoid_hist_rows.shape[1]).astype(np.int32)
        return True

    def _write_map_cell(self, h5, key: str, row: int, col: int, value) -> bool:
        if h5 is None:
            return False
        ds = None
        try:
            if key in h5:
                ds = h5[key]
        except Exception:
            ds = None
        if ds is None:
            ds = hdf5_io.read_dataset(h5, key, default=None)
        if ds is None:
            return False
        try:
            ds[row, col] = value
            return True
        except Exception:
            try:
                arr = np.array(ds)
                if arr.ndim != 2:
                    return False
                arr[row, col] = value
                hdf5_io.write_dataset(h5, key, arr)
                return True
            except Exception:
                return False

    def derive_environment_refugia(self, ref_length_mm: float | None = None) -> bool:
        """Create `environment/refugia` (binary) using fatigued station-holding criterion.

        Refugia cells are those where water speed <= max_s_U_fatigued (BL/s) converted
        to m/s using a reference fish length. This avoids per-agent maps while still
        encoding the canonical definition.
        """
        h5 = hdf5_io.get_hdf5_obj(self)
        if h5 is None:
            return False
        existing = hdf5_io.read_dataset(h5, 'environment/refugia', default=None)
        if existing is not None:
            self._refugia_derived = True
            return True

        vel_mag = hdf5_io.read_dataset(h5, 'environment/vel_mag', default=None)
        if vel_mag is None:
            vel_x = hdf5_io.read_dataset(h5, 'environment/vel_x', default=None)
            vel_y = hdf5_io.read_dataset(h5, 'environment/vel_y', default=None)
            if vel_x is None or vel_y is None:
                return False
            vel_mag_arr = np.sqrt(np.asarray(vel_x, dtype=float) ** 2 + np.asarray(vel_y, dtype=float) ** 2)
        else:
            vel_mag_arr = np.asarray(vel_mag, dtype=float)

        if vel_mag_arr.ndim != 2 or vel_mag_arr.size <= 1:
            return False

        if ref_length_mm is None:
            ref_length_mm = getattr(self, 'refugia_ref_length_mm', None)
        try:
            ref_length_mm = float(ref_length_mm)
        except Exception:
            ref_length_mm = float(getattr(self, 'refugia_ref_length_mm', 500.0))
        if not np.isfinite(ref_length_mm) or ref_length_mm <= 0:
            ref_length_mm = 500.0

        try:
            max_s_bl_s = float(np.nanmedian(np.asarray(getattr(self, 'max_s_U_fatigued', self.max_s_U), dtype=float)))
        except Exception:
            max_s_bl_s = 2.77
        thresh_m_s = max_s_bl_s * (ref_length_mm / 1000.0)

        refugia = (vel_mag_arr <= thresh_m_s).astype(np.uint8)
        hdf5_io.write_dataset(h5, 'environment/refugia', refugia)
        self._refugia_derived = True
        return True

    def update_avoid_memory(self, t: float) -> bool:
        """Update avoid memory at current positions.

        Non-debug default: update sparse per-agent visit history (fast, no per-agent HDF5 writes).
        Debug/explicit: also write into per-agent HDF5 rasters under `memory/<i>`.
        """
        h5 = hdf5_io.get_hdf5_obj(self)
        if h5 is None:
            return False

        persist_dense = bool(getattr(self, 'persist_avoid_memory_hdf5', False) or getattr(self, 'debug_behavior', False))

        # ensure transform is defined; only create dense datasets when persisting
        if getattr(self, 'mental_map_transform', None) is None:
            ok = self.initialize_mental_map(create_datasets=persist_dense)
            if not ok:
                return False
        try:
            rows, cols = utils.geo_to_pixel(self.X, self.Y, self.mental_map_transform)
            rows = np.atleast_1d(rows).astype(int)
            cols = np.atleast_1d(cols).astype(int)
        except Exception:
            return False

        # sparse history update (preferred)
        if bool(getattr(self, 'use_sparse_avoid_memory', True)):
            if self._avoid_map_shape is None:
                # best-effort infer from existing datasets or initialize_mental_map metadata
                try:
                    if hasattr(self, '_avoid_map_shape') and self._avoid_map_shape is not None:
                        pass
                except Exception:
                    pass
            if not self.ensure_avoid_history():
                return False
            n = int(self.num_agents)
            idx = np.arange(n, dtype=int)
            pos = np.asarray(self.avoid_hist_pos, dtype=int)
            k = int(self.avoid_hist_rows.shape[1])
            last_pos = (pos - 1) % k
            last_r = np.asarray(self.avoid_hist_rows[idx, last_pos], dtype=int)
            last_c = np.asarray(self.avoid_hist_cols[idx, last_pos], dtype=int)
            changed = (rows != last_r) | (cols != last_c)
            # validity within avoid grid when known
            if self._avoid_map_shape is not None:
                ah, aw = self._avoid_map_shape
                valid = (rows >= 0) & (cols >= 0) & (rows < int(ah)) & (cols < int(aw))
            else:
                valid = (rows >= 0) & (cols >= 0)
            mask = changed & valid
            if np.any(mask):
                sel = idx[mask]
                psel = pos[mask]
                self.avoid_hist_rows[sel, psel] = rows[mask].astype(np.int16)
                self.avoid_hist_cols[sel, psel] = cols[mask].astype(np.int16)
                self.avoid_hist_t[sel, psel] = float(t)
                pos2 = pos.copy()
                pos2[mask] = (pos2[mask] + 1) % k
                self.avoid_hist_pos = pos2.astype(np.int32)

        # Optional dense HDF5 write (debug / explicit)
        if persist_dense:
            # ensure datasets exist
            if getattr(self, 'mental_map_transform', None) is None:
                return False
            if not self.initialize_mental_map(create_datasets=True):
                return False
            for i in range(int(self.num_agents)):
                r = int(rows[i])
                c = int(cols[i])
                key = f'memory/{i}'
                ds0 = hdf5_io.read_dataset(h5, key, default=None)
                if ds0 is None:
                    continue
                try:
                    nrows, ncols = np.asarray(ds0).shape
                except Exception:
                    continue
                if r < 0 or c < 0 or r >= nrows or c >= ncols:
                    continue
                self._write_map_cell(h5, key, r, c, float(t))
        return True

    def timestep(self, t, dt, g=None, pid_controller=None):
        # Advance time and run a single simulation timestep integrating
        # behavior -> fatigue -> movement -> write outputs.
        self.cumulative_time += dt

        # Standardize a per-step marker used by diagnostics/caches.
        try:
            self.current_step = int(t)
        except Exception:
            self.current_step = t

        # Reset per-timestep environment-sample cache: X/Y are stable until after
        # movement updates later in this method.
        try:
            if getattr(self, 'cache_env_samples', True):
                self._env_sample_cache_step = int(getattr(self, 'current_step', t))
                self._env_sample_cache_gen = 0
                self._env_sample_cache = {}
                self._env_pixel_cache = {}
        except Exception:
            pass

        # keep previous positions for velocity calculations
        self.prev_X = self.X.copy()
        self.prev_Y = self.Y.copy()

        # ensure a pid controller is available
        pid = pid_controller or self.pid_controller

        # mask of agents able to move
        mask = np.where(self.dead == 0, True, False)

        # --- environment sampling: populate water velocities / depth at current positions
        # movement and fatigue modules treat `x_vel/y_vel` as water velocities.
        depth = self.sample_environment(getattr(self, 'depth_rast_transform', None), 'depth')
        self.depth = np.asarray(depth, dtype=np.float32)

        tx = getattr(self, 'vel_x_rast_transform', None) or getattr(self, 'depth_rast_transform', None)
        ty = getattr(self, 'vel_y_rast_transform', None) or getattr(self, 'depth_rast_transform', None)
        x_vel = self.sample_environment(tx, 'vel_x')
        y_vel = self.sample_environment(ty, 'vel_y')
        self.x_vel = np.where(np.isnan(x_vel), 0.0, x_vel).astype(np.float32)
        self.y_vel = np.where(np.isnan(y_vel), 0.0, y_vel).astype(np.float32)

        # ensure refugia mask exists when enabled (computed once per run)
        if getattr(self, 'auto_derive_refugia', False) and not getattr(self, '_refugia_derived', False):
            try:
                self.derive_environment_refugia()
            except Exception:
                pass

        # --- neighbor finding: populate CSR neighbors and/or nearest-neighbor fields
        # Keep this work conditional so profiling / acceptance runs that isolate
        # non-schooling cues don't pay O(N log N) neighbor costs.
        try:
            n_agents = int(getattr(self, 'num_agents', 0) or 0)
        except Exception:
            n_agents = 0
        try:
            debug_behavior = bool(getattr(self, 'debug_behavior', False))
            build_buffers = bool(getattr(self, 'build_agents_within_buffers', False)) or debug_behavior
        except Exception:
            debug_behavior = False
            build_buffers = False

        # Determine whether schooling/collision cues are active in this step.
        # In normal runs `test_weights` is absent and we assume neighbors are needed.
        tw = getattr(self, 'test_weights', None)
        need_alignment = True
        need_cohesion = True
        need_collision = True
        if isinstance(tw, dict) and tw:
            def _nz(k: str) -> bool:
                try:
                    return float(tw.get(k, 0.0)) != 0.0
                except Exception:
                    return False
            need_alignment = _nz('alignment')
            need_cohesion = _nz('cohesion')
            need_collision = _nz('collision')

        need_neighbor_graph = build_buffers or need_alignment or need_cohesion
        need_nearest = need_collision

        if cKDTree is not None and n_agents > 0 and (need_neighbor_graph or need_nearest):
            try:
                # Throttle neighbor rebuilds: default every ~2 seconds.
                # Always build on first use.
                step_i = None
                try:
                    step_i = int(getattr(self, 'current_step', t))
                except Exception:
                    try:
                        step_i = int(t)
                    except Exception:
                        step_i = None

                interval_steps = int(getattr(self, 'neighbor_update_interval_steps', 0) or 0)
                if interval_steps <= 0:
                    try:
                        seconds = float(getattr(self, 'neighbor_update_seconds', 2.0))
                    except Exception:
                        seconds = 2.0
                    if not np.isfinite(seconds) or seconds <= 0.0:
                        seconds = 2.0
                    try:
                        interval_steps = max(1, int(round(seconds / float(dt))))
                    except Exception:
                        interval_steps = 1

                last_step = getattr(self, '_neighbor_last_build_step', None)
                have_graph = getattr(self, 'neighbors_offsets', None) is not None and getattr(self, 'neighbors_indices', None) is not None
                have_nearest = getattr(self, 'closest_agent', None) is not None and getattr(self, 'nearest_neighbor_distance', None) is not None
                need_build_now = False
                if not ((need_neighbor_graph and have_graph) or (need_nearest and have_nearest)):
                    need_build_now = True
                elif step_i is None or last_step is None:
                    need_build_now = True
                else:
                    try:
                        need_build_now = (int(step_i) - int(last_step)) >= int(interval_steps)
                    except Exception:
                        need_build_now = True

                if not need_build_now:
                    # Keep previous neighbor fields; skip rebuild work this step.
                    raise StopIteration()

                # Reuse a scratch points array to reduce per-step allocations.
                pts = getattr(self, '_neighbor_pts_scratch', None)
                if not isinstance(pts, np.ndarray) or pts.shape != (n_agents, 2):
                    pts = np.empty((n_agents, 2), dtype=np.float64)
                    self._neighbor_pts_scratch = pts
                # Assigning into a float64 buffer performs any needed casting
                # without allocating float64 copies of X/Y.
                pts[:, 0] = np.asarray(self.X).reshape((-1,))
                pts[:, 1] = np.asarray(self.Y).reshape((-1,))
                tree = cKDTree(pts)

                # Allow multi-threaded queries when available (SciPy `workers`).
                try:
                    workers = int(getattr(self, 'neighbor_workers', 1) or 1)
                except Exception:
                    workers = 1
                if workers == 0:
                    workers = 1

                if need_neighbor_graph:
                    # buffer radius in meters (use a simulation attribute or default)
                    radius = self._neighbor_radius_m()

                    # Prefer unsorted neighbor lists (we don't require stable ordering).
                    try:
                        neighbors_lists = tree.query_ball_point(pts, r=radius, return_sorted=False, workers=workers)
                    except TypeError:
                        try:
                            neighbors_lists = tree.query_ball_point(pts, r=radius, workers=workers)
                        except TypeError:
                            neighbors_lists = tree.query_ball_point(pts, r=radius)

                    lens = np.fromiter((len(lst) for lst in neighbors_lists), dtype=np.int32, count=n_agents)
                    total = int(lens.sum())
                    neighbors_offsets = np.empty(n_agents + 1, dtype=np.int64)
                    neighbors_offsets[0] = 0
                    neighbors_indices = np.empty(total, dtype=np.int32)

                    pos = 0
                    for i, lst in enumerate(neighbors_lists):
                        k = int(len(lst))
                        end = pos + k
                        if k > 0:
                            neighbors_indices[pos:end] = lst
                        pos = end
                        neighbors_offsets[i + 1] = pos
                    counts = lens.astype(np.int32, copy=False)

                    self.neighbors_offsets = neighbors_offsets
                    self.neighbors_indices = neighbors_indices
                    # Count neighbors excluding the self entry (assumes each list includes self).
                    self.neighbor_counts = np.maximum(0, counts - 1).astype(np.int32, copy=False)

                    # Legacy compatibility: only build per-agent buffers when requested.
                    if build_buffers:
                        self.agents_within_buffers = [
                            neighbors_indices[neighbors_offsets[i] : neighbors_offsets[i + 1]]
                            for i in range(n_agents)
                        ]

                if need_nearest:
                    # nearest neighbor excluding self (used by collision cue)
                    try:
                        distances, indices = tree.query(pts, k=2, workers=workers)
                    except TypeError:
                        distances, indices = tree.query(pts, k=2)
                    # distances[:,0] == 0 (self), so take 1
                    nearest = np.where(np.isfinite(distances[:, 1]), indices[:, 1], np.nan)
                    nearest_d = np.where(np.isfinite(distances[:, 1]), distances[:, 1], np.nan)
                    self.closest_agent = nearest
                    self.nearest_neighbor_distance = nearest_d

                try:
                    self._neighbor_last_build_step = step_i
                except Exception:
                    pass
            except StopIteration:
                # Normal control flow: neighbor update not due yet.
                pass
            except Exception:
                # Leave neighbor defaults in place
                pass

        # instantiate per-timestep helpers (reuse instances to avoid per-step allocation)
        behavior = getattr(self, '_behavior', None)
        if behavior is None:
            behavior = behavior_mod.behavior(dt, self)
            self._behavior = behavior
        else:
            try:
                behavior.dt = dt
            except Exception:
                pass

        fatigue = getattr(self, '_fatigue', None)
        if fatigue is None:
            try:
                fatigue = fatigue_mod.fatigue(t, dt, self)
                self._fatigue = fatigue
            except Exception:
                fatigue = None
        else:
            try:
                fatigue.t = t
                fatigue.dt = dt
            except Exception:
                pass

        movement = getattr(self, '_movement', None)
        if movement is None:
            try:
                movement = movement_mod.movement(self)
                self._movement = movement
            except Exception:
                movement = None

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

        def _movement_call(label, func, *args, default=None):
            try:
                return func(*args)
            except Exception as e:
                if getattr(self, 'debug_freq', False):
                    try:
                        logging.getLogger(__name__).exception('%s exception: %s', label, e)
                    except Exception:
                        pass
                return default

        # calculate movement-related quantities with finer-grained diagnostics
        dxdy = np.zeros((self.num_agents, 2), dtype=np.float32)
        if movement is not None:
            _movement_call('movement.frequency', movement.frequency, mask, t, dt)
            _movement_call('movement.thrust_fun', movement.thrust_fun, mask, t, dt)
            _movement_call('movement.drag_fun', movement.drag_fun, mask, t, dt)
            dxdy = _movement_call('movement.swim', movement.swim, t, dt, pid or pid_controller, mask, default=dxdy)

        # If debugging is enabled, print compact diagnostics to help trace zero-values
        if getattr(self, 'debug_freq', False):
            try:
                logging.getLogger(__name__).debug('DEBUG movement: Hz[:10]=%s', self.Hz[:10])
                logging.getLogger(__name__).debug('DEBUG movement: thrust[:5]=%s', self.thrust[:5])
                logging.getLogger(__name__).debug('DEBUG movement: drag[:5]=%s', self.drag[:5])
                logging.getLogger(__name__).debug('DEBUG movement: length[:5]=%s', self.length[:5])
                logging.getLogger(__name__).debug('DEBUG movement: weight[:5]=%s', self.weight[:5])
                logging.getLogger(__name__).debug('DEBUG movement: swim_behav[:10]=%s', self.swim_behav[:10])
                logging.getLogger(__name__).debug('DEBUG movement: is_stuck[:10]=%s', self.is_stuck[:10])
                logging.getLogger(__name__).debug('DEBUG movement: prev_Hz[:10]=%s', self.prev_Hz[:10])
            except Exception:
                pass

        # apply movement (support both (N,2) and (N,) displacements)
        dxdy = np.asarray(dxdy)
        if dxdy.shape == (self.num_agents, 2):
            self.X = self.X + dxdy[:, 0]
            self.Y = self.Y + dxdy[:, 1]
        else:
            d = dxdy.reshape((-1,))
            self.X = self.X + d
            self.Y = self.Y + d

        # X/Y changed; invalidate any cached samples from the pre-movement state.
        try:
            if getattr(self, 'cache_env_samples', True) and isinstance(getattr(self, '_env_sample_cache', None), dict):
                self._env_sample_cache_gen = int(getattr(self, '_env_sample_cache_gen', 0) or 0) + 1
                self._env_sample_cache = {}
                self._env_pixel_cache = {}
        except Exception:
            pass

        # update fish kinematics (do not overwrite water velocity fields)
        try:
            self.fish_x_vel = np.asarray((self.X - self.prev_X) / dt, dtype=np.float32)
            self.fish_y_vel = np.asarray((self.Y - self.prev_Y) / dt, dtype=np.float32)
            self.sog = np.asarray(np.sqrt(self.fish_x_vel**2 + self.fish_y_vel**2), dtype=np.float32)
        except Exception:
            pass

        # write minimal outputs back to HDF5 for downstream consumers
        # update avoid memory after movement so next steps can repel from recently visited areas
        try:
            self.update_avoid_memory(t)
        except Exception:
            pass
        disable_writes = bool(getattr(self, 'disable_output_writes', False) or getattr(self, 'disable_hdf_writes', False))
        if not disable_writes:
            hdf5_io.write_dataset(self.db, 'X', self.X)
            hdf5_io.write_dataset(self.db, 'Y', self.Y)
            hdf5_io.write_dataset(self.db, 'prev_X', self.prev_X)
            hdf5_io.write_dataset(self.db, 'prev_Y', self.prev_Y)

        # write per-timestep slices into time-indexed agent_data arrays
        h5 = hdf5_io.get_hdf5_obj(self)
        ts = int(max(0, min(int(self.cumulative_time) - 1, self.num_timesteps - 1)))
        try:
            t_int = int(t)
            if 0 <= t_int < int(self.num_timesteps):
                ts = t_int
        except Exception:
            pass

        write_frequency = int(getattr(self, 'write_frequency', 1) or 0)
        do_timeseries_write = (not disable_writes) and write_frequency > 0 and (ts % write_frequency == 0)
        if do_timeseries_write:
            tracked = ('agent_data/X', 'agent_data/Y', 'agent_data/prev_X', 'agent_data/prev_Y', 'agent_data/ideal_sog', 'agent_data/Hz')
            for key in tracked:
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
                ok = hdf5_io.write_timeseries_step(h5, key, ts, val)
                if not ok:
                    logging.getLogger(__name__).debug('Dataset %s missing/unwritable; skipping timestep write', key)

        # flush when supported (best-effort)
        flush_frequency_raw = getattr(self, 'flush_frequency', None)
        if flush_frequency_raw is None:
            flush_frequency_raw = getattr(self, 'write_frequency', 1)
        flush_frequency = int(flush_frequency_raw or 0)
        do_flush = (not disable_writes) and flush_frequency > 0 and (ts % flush_frequency == 0)
        if do_flush and hasattr(self.db, 'flush'):
            try:
                self.db.flush()
            except Exception:
                pass

        return True

    def sample_environment(self, transform, raster_name):
        """Sample `environment/<raster_name>` at agent positions and return array of values.

        This thin wrapper uses `geo_to_pixel` to convert agent `X,Y` into pixel indices
        using the provided `transform`, then reads the environment dataset via `hdf5_io`.
        Returns a 1-D numpy array of length `num_agents` filled with np.nan for out-of-bounds.
        """
        h5 = hdf5_io.get_hdf5_obj(self)
        if h5 is None or transform is None:
            return np.full(self.num_agents, np.nan)

        # Per-timestep cache (dedupe repeated sampling across multiple cues).
        try:
            if getattr(self, 'cache_env_samples', True):
                step_i = int(getattr(self, 'current_step', -1))
                cache_step = getattr(self, '_env_sample_cache_step', None)
                cache = getattr(self, '_env_sample_cache', None)
                if cache_step is not None and int(cache_step) == step_i and isinstance(cache, dict):
                    k = self._env_sample_cache_key(transform, raster_name)
                    if k in cache:
                        return cache[k]
        except Exception:
            pass

        ds_arr = self.get_cached_dataset(f'environment/{raster_name}', default=None)
        if ds_arr is None:
            return np.full(self.num_agents, np.nan)
        ds_arr = np.asarray(ds_arr)

        rows = None
        cols = None
        try:
            if getattr(self, 'cache_env_samples', True):
                pcache = getattr(self, '_env_pixel_cache', None)
                if isinstance(pcache, dict):
                    pk = self._env_pixel_cache_key(transform)
                    if pk in pcache:
                        rows, cols = pcache[pk]
        except Exception:
            rows = None
            cols = None

        if rows is None or cols is None:
            try:
                rows, cols = utils.geo_to_pixel(self.X, self.Y, transform)
            except Exception:
                return np.full(self.num_agents, np.nan)
            try:
                if getattr(self, 'cache_env_samples', True):
                    pcache = getattr(self, '_env_pixel_cache', None)
                    if isinstance(pcache, dict):
                        pk = self._env_pixel_cache_key(transform)
                        pcache[pk] = (rows, cols)
            except Exception:
                pass

        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        valid = (rows >= 0) & (cols >= 0) & (rows < ds_arr.shape[0]) & (cols < ds_arr.shape[1])
        out = np.full(self.num_agents, np.nan)
        if np.any(valid):
            try:
                out[valid] = ds_arr[rows[valid], cols[valid]]
            except Exception:
                for i in np.where(valid)[0]:
                    try:
                        out[i] = ds_arr[rows[i], cols[i]]
                    except Exception:
                        out[i] = np.nan

        if getattr(self, 'debug_env', False):
            try:
                logging.getLogger(__name__).debug('sample_environment debug: raster=%s transform=%s', raster_name, transform)
                logging.getLogger(__name__).debug('rows sample (first 5): %s cols sample (first 5): %s', rows[:5], cols[:5])
                logging.getLogger(__name__).debug('valid count: %s', int(np.sum(valid)))
            except Exception:
                pass

        try:
            if getattr(self, 'cache_env_samples', True):
                step_i = int(getattr(self, 'current_step', -1))
                cache_step = getattr(self, '_env_sample_cache_step', None)
                cache = getattr(self, '_env_sample_cache', None)
                if cache_step is not None and int(cache_step) == step_i and isinstance(cache, dict):
                    k = self._env_sample_cache_key(transform, raster_name)
                    cache[k] = out
        except Exception:
            pass
        return out

    def get_cached_dataset(self, key: str, default=None):
        """Return a cached numpy array for static datasets (primarily environment/*).

        For h5py-backed runs, this avoids re-reading large rasters each timestep.
        Only keys under `environment/` are cached to prevent stale writes for
        mutable datasets.
        """
        try:
            key = str(key)
        except Exception:
            return default
        cacheable = key.startswith('environment/')
        if not cacheable:
            return hdf5_io.read_dataset(hdf5_io.get_hdf5_obj(self), key, default=default)
        try:
            cache = getattr(self, '_dataset_cache', None)
            if isinstance(cache, dict) and key in cache:
                return cache[key]
        except Exception:
            cache = None
        h5 = hdf5_io.get_hdf5_obj(self)
        val = hdf5_io.read_dataset(h5, key, default=default)
        try:
            if not isinstance(getattr(self, '_dataset_cache', None), dict):
                self._dataset_cache = {}
            self._dataset_cache[key] = val
        except Exception:
            pass
        return val

    def run(self, model_name=None, n=1, dt=1.0, video=False, k_p=None, k_i=None, k_d=None, return_status: bool = False, video_hook=None, viewer: bool = False, viewer_blocking: bool = False, viewer_live: bool = False, viewer_host: str = '127.0.0.1', viewer_port: int = 50007, viewer_stream_raw: bool = False, viewer_fps: float = 20.0):
        # Enhanced run loop with PID plumbing, write frequency, optional video hook,
        # and optional real-time viewer. Backwards compatible: original signature still works.
        write_frequency = 1
        video_hook = None

        # Accept a PID controller instance instead of scalar gains via model_name kw
        controller = None
        if isinstance(model_name, dict) and 'pid_controller' in model_name:
            controller = model_name.pop('pid_controller')

        # If scalar gains provided, set controller gains (if present)
        if controller is None:
            controller = self.pid_controller

        if controller is not None and k_p is not None:
            try:
                controller.k_p = np.array([k_p]) if np.isscalar(k_p) else np.array(k_p)
                if k_i is not None:
                    controller.k_i = np.array([k_i]) if np.isscalar(k_i) else np.array(k_i)
                if k_d is not None:
                    controller.k_d = np.array([k_d]) if np.isscalar(k_d) else np.array(k_d)
            except Exception:
                pass

        status = {'steps': 0, 'errors': [], 'video_frames': 0}

        # Optionally launch the realtime viewer as a subprocess that reads
        # the HDF5 database written by this simulation. If `viewer_blocking` is
        # True the run will block until the viewer exits; otherwise the viewer
        # runs in parallel.
        viewer_proc = None
        # live server socket used to stream frames to a connected viewer client
        live_server = None
        client_conn = None
        if viewer_live:
            try:
                import socket
                live_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                live_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                live_server.bind((viewer_host, int(viewer_port)))
                live_server.listen(1)
                # set to non-blocking so accept can be polled
                live_server.setblocking(False)
            except Exception as e:
                status['errors'].append(f'viewer_live_bind_error:{e}')
        if viewer:
            try:
                import subprocess
                viewer_cmd = [
                    sys.executable,
                    "-m",
                    "emergent.salmon_abm.realtime_viewer",
                    self.db_path,
                ]
                # Launch detached on Windows so it doesn't inherit std handles
                viewer_proc = subprocess.Popen(viewer_cmd, creationflags=0)
            except Exception as e:
                status['errors'].append(f'viewer_launch_error:{e}')
        for i in range(n):
            try:
                self.timestep(i, dt, pid_controller=controller)
                status['steps'] += 1
                # optional video hook called after each timestep
                if video_hook is not None:
                    try:
                        video_hook(self, i)
                        status['video_frames'] += 1
                    except Exception as e:
                        status['errors'].append(f'video_hook_error:{e}')
                # Live viewer: accept a client and stream the current positions frame
                if viewer_live and live_server is not None:
                    try:
                        # accept a single client if not connected
                        if client_conn is None:
                            try:
                                conn, addr = live_server.accept()
                                conn.setblocking(True)
                                client_conn = conn
                            except BlockingIOError:
                                conn = None
                        if client_conn is not None:
                            # send current frame positions as either raw float32 or numpy .npy
                            try:
                                import io
                                import struct
                                # build frame as (N,2) float32 array
                                xs = getattr(self, 'X', None)
                                ys = getattr(self, 'Y', None)
                                if xs is not None and ys is not None:
                                    frame = np.vstack((xs, ys)).T.astype(np.float32)
                                else:
                                    frame = np.zeros((self.num_agents, 2), dtype=np.float32)
                                if viewer_stream_raw:
                                    # raw protocol: 'R' + 4-byte length + payload
                                    payload = frame.astype(np.float32).tobytes()
                                    client_conn.sendall(b'R' + struct.pack('!I', len(payload)) + payload)
                                else:
                                    buf = io.BytesIO()
                                    np.save(buf, frame)
                                    data = buf.getvalue()
                                    client_conn.sendall(struct.pack('!I', len(data)))
                                    client_conn.sendall(data)
                            except Exception:
                                try:
                                    client_conn.close()
                                except Exception:
                                    pass
                                client_conn = None
                    except Exception:
                        pass
            except Exception as e:
                status['errors'].append(str(e))
                # continue running unless unrecoverable
                continue

        # flush and close viewer process if requested
        if hasattr(self.db, 'flush'):
            try:
                self.db.flush()
            except Exception:
                pass

        # If we launched the viewer and the user wants blocking behavior, wait
        if viewer and viewer_proc is not None:
            try:
                if viewer_blocking:
                    viewer_proc.wait()
            except Exception:
                pass

        # preserve legacy return value for backwards compatibility
        self.last_run_status = status
        if return_status or video:
            return status
        return True

    def close(self):
        # close HDF5 and optionally remove temporary DB file if it was created internally
        beh = getattr(self, "_behavior", None)
        stop_thread = getattr(beh, "_stop_diag_thread", None) if beh is not None else None
        if callable(stop_thread):
            try:
                stop_thread()
            except Exception:
                pass

        db = getattr(self, "db", None)
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

        if getattr(self, "_created_db_file", False):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        return True


__all__ = ['simulation']
