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
                    arrX = hdf5_io.read_dataset(self.db, 'agent_data/X', default=None)
                    arrY = hdf5_io.read_dataset(self.db, 'agent_data/Y', default=None)
                    if arrX is not None:
                        arrX[:, 0] = self.X
                        hdf5_io.write_dataset(self.db, 'agent_data/X', arrX)
                    if arrY is not None:
                        arrY[:, 0] = self.Y
                        hdf5_io.write_dataset(self.db, 'agent_data/Y', arrY)
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
        for ef in self.env_files:
            _ = io.enviro_import(ef)

        # ensure minimal environment placeholders exist so downstream modules
        # that read environment/* will have something to sample in unit tests
        hdf5_io.create_environment_placeholders(self.db)

        # Create x/y coordinate grids and attach simple affine transforms so
        # behavior and sampling helpers can map geo <-> pixel indices.
        try:
            h5 = hdf5_io.get_hdf5_obj(self)
            depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=np.zeros((1, 1)))
            nrows, ncols = depth_ds.shape
            # Prefer to use an attached raster transform to compute real-world
            # x_coords/y_coords (pixel -> geo). If no transform is available
            # fall back to simple index-based coordinates so tests still run.
            try:
                transform = getattr(self, 'depth_rast_transform', None)
                if transform is not None and not callable(transform):
                    # Transform expected as a 6-tuple (a, b, c, d, e, f)
                    # where (col, row) -> (x, y) via affine: x = a*col + b*row + c
                    # and y = d*col + e*row + f
                    cols = np.arange(ncols, dtype=float)
                    rows = np.arange(nrows, dtype=float)
                    col_indices, row_indices = np.meshgrid(cols, rows)
                    a, b, c, d, e, f = transform
                    x_coords = a * col_indices + b * row_indices + c
                    y_coords = d * col_indices + e * row_indices + f
                else:
                    # fallback to index-based coordinates
                    x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
                    y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
            except Exception:
                x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
                y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
            # write both environment/ prefixed and top-level keys for compatibility
            hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
            hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
            hdf5_io.write_dataset(h5, 'x_coords', x_coords)
            hdf5_io.write_dataset(h5, 'y_coords', y_coords)
            # attach simple identity-like affine transforms (a,b,c,d,e,f)
            # mapping pixel -> geo as x=col, y=row
            self.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            self.vel_mag_rast_transform = self.depth_rast_transform
            self.vel_dir_rast_transform = self.depth_rast_transform
            self.refugia_map_transform = self.depth_rast_transform
        except Exception:
            # tolerate any failures here; behavior will be more limited but simulation can still run
            self.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            self.vel_mag_rast_transform = self.depth_rast_transform
            self.vel_dir_rast_transform = self.depth_rast_transform
            self.refugia_map_transform = self.depth_rast_transform

        # small helpers: construct movement and behavior helpers now
        self._movement = movement_mod.movement(self)
        self._behavior = behavior_mod.behavior(1.0, self)
        self._fatigue = None
        # ensure attributes expected by movement/behavior exist with sensible defaults
        try:
            self.pid_tuning = pid_tuning
        except Exception:
            self.pid_tuning = False
        try:
            # wave_drag used by drag calculations
            self.wave_drag = np.ones(self.num_agents, dtype=float)
        except Exception:
            self.wave_drag = np.ones(self.num_agents)
        try:
            # Hz used by thrust calculations
            self.Hz = np.zeros(self.num_agents, dtype=float)
        except Exception:
            self.Hz = np.zeros(self.num_agents)
        # expose drag_coeff on simulation so movement can call it
        try:
            self.drag_coeff = self._movement.drag_coeff
        except Exception:
            # fallback to a simple interpolation if movement helper not available
            self.drag_coeff = lambda reynolds: np.interp(reynolds, [2.5e4, 5.0e4, 7.4e4, 9.9e4, 1.2e5, 1.5e5, 1.7e5, 2.0e5], [0.23, 0.19, 0.15, 0.14, 0.12, 0.12, 0.11, 0.10])

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

        # calculate movement-related quantities with finer-grained diagnostics
        dxdy = np.zeros((self.num_agents, 2), dtype=np.float32)
        if movement is not None:
            # frequency
            try:
                movement.frequency(mask, t, dt)
            except Exception as e:
                if getattr(self, 'debug_freq', False):
                    import traceback
                    print('movement.frequency exception:', e)
                    traceback.print_exc()
            # thrust
            try:
                movement.thrust_fun(mask, t, dt)
            except Exception as e:
                if getattr(self, 'debug_freq', False):
                    import traceback
                    print('movement.thrust_fun exception:', e)
                    traceback.print_exc()
            # drag
            try:
                movement.drag_fun(mask, t, dt)
            except Exception as e:
                if getattr(self, 'debug_freq', False):
                    import traceback
                    print('movement.drag_fun exception:', e)
                    traceback.print_exc()
            # swim (returns displacement)
            try:
                dxdy = movement.swim(t, dt, pid or pid_controller, mask)
            except Exception as e:
                if getattr(self, 'debug_freq', False):
                    import traceback
                    print('movement.swim exception:', e)
                    traceback.print_exc()
        else:
            dxdy = np.zeros((self.num_agents, 2), dtype=np.float32)

        # If debugging is enabled, print compact diagnostics to help trace zero-values
        if getattr(self, 'debug_freq', False):
            try:
                print('DEBUG movement: Hz[:10]=', self.Hz[:10])
                print('DEBUG movement: thrust[:5]=', self.thrust[:5])
                print('DEBUG movement: drag[:5]=', self.drag[:5])
                print('DEBUG movement: length[:5]=', self.length[:5])
                print('DEBUG movement: weight[:5]=', self.weight[:5])
                print('DEBUG movement: swim_behav[:10]=', self.swim_behav[:10])
                print('DEBUG movement: is_stuck[:10]=', self.is_stuck[:10])
                print('DEBUG movement: prev_Hz[:10]=', self.prev_Hz[:10])
            except Exception:
                pass

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

    def sample_environment(self, transform, raster_name):
        """Sample `environment/<raster_name>` at agent positions and return array of values.

        This thin wrapper uses `geo_to_pixel` to convert agent `X,Y` into pixel indices
        using the provided `transform`, then reads the environment dataset via `hdf5_io`.
        Returns a 1-D numpy array of length `num_agents` filled with np.nan for out-of-bounds.
        """
        try:
            from emergent.salmon_abm.utils import geo_to_pixel
            h5 = hdf5_io.get_hdf5_obj(self)
            ds = hdf5_io.read_dataset(h5, f'environment/{raster_name}', default=None)
            if ds is None:
                return np.full(self.num_agents, np.nan)
            # ensure ds is a numpy array to avoid h5py advanced-index restrictions
            try:
                ds_arr = np.array(ds)
            except Exception:
                ds_arr = ds

            # geo_to_pixel accepts arrays and returns (rows, cols)
            rows, cols = geo_to_pixel(self.X, self.Y, transform)
            # ensure integer indices and bounds
            rows = np.asarray(rows, dtype=int)
            cols = np.asarray(cols, dtype=int)
            valid = (rows >= 0) & (cols >= 0) & (rows < ds.shape[0]) & (cols < ds.shape[1])
            out = np.full(self.num_agents, np.nan)
            if np.any(valid):
                try:
                    out[valid] = ds_arr[rows[valid], cols[valid]]
                except Exception:
                    # fallback: loop assign to avoid advanced indexing issues
                    for i in np.where(valid)[0]:
                        try:
                            out[i] = ds_arr[rows[i], cols[i]]
                        except Exception:
                            out[i] = np.nan
            # optional debug prints
            if getattr(self, 'debug_env', False):
                try:
                    print('sample_environment debug:', raster_name, 'transform=', transform)
                    print('rows sample (first 5):', rows[:5], 'cols sample (first 5):', cols[:5])
                    print('valid count:', int(np.sum(valid)))
                except Exception:
                    pass
            return out
        except Exception:
            return np.full(self.num_agents, np.nan)

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
                import subprocess, shlex
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
        try:
            if hasattr(self.db, 'flush'):
                try:
                    self.db.flush()
                except Exception:
                    pass
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
