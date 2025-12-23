"""Run a 100-agent simulation for 10 timesteps using real rasters from data/salmon_abm."""
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Run instrumented sim with real env rasters')
parser.add_argument('--nsteps', type=int, default=10, help='number of timesteps to run')
parser.add_argument('--nagents', type=int, default=100, help='number of agents to simulate')
args = parser.parse_args()

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
# Build env files list: depth, vel_x, vel_y, vel_mag, vel_dir
env_files = [os.path.join(data_dir, fname) for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']]
start_poly = os.path.join(data_dir, 'start_loc_river_right.shp')

sim = simulation(model_dir='.', model_name='real_probe', crs=None, basin='test', water_temp=10, start_polygon=start_poly, env_files=env_files, longitudinal_profile=None, fish_length=200.0, num_timesteps=args.nsteps, num_agents=args.nagents)

# enable debug flags for diagnostics
sim.debug_env = True
sim.debug_freq = True
# enable fine-grained frequency probe CSV output
sim.debug_freq_probe = True

# Import rasters into sim.db using io.enviro_import if available
for ef in env_files:
    try:
        arr, transform, crs = io.enviro_import(ef)
        # write arrays in environment/ prefix keyed by base filename without extension
        key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
        hdf5_io.write_dataset(sim.db, key, arr)
        # also write x/y coords if possible
    except Exception as e:
        print('env import failed for', ef, e)

# write x/y coords and transforms consistent with rasters
# attempt to read back one raster to get shape
depth = hdf5_io.read_dataset(sim.db, 'environment/depth')
if depth is not None:
    nrows, ncols = depth.shape
    # If io.enviro_import returned a transform for the rasters above, use it
    # to compute geo x/y coordinates for each pixel. Otherwise fall back to
    # index-based coordinates.
    try:
        # use depth transform if available
        depth_entry = io.enviro_import(env_files[0])
        _, depth_transform, _ = depth_entry
    except Exception:
        depth_transform = getattr(sim, 'depth_rast_transform', None)

    def _affine_to_tuple(transform):
        """Return a 6-tuple (a,b,c,d,e,f) from common transform types.

        Supports rasterio Affine objects (attributes a,b,c,d,e,f) or
        any iterable with at least 6 elements. Returns None if not possible.
        """
        if transform is None:
            return None
        # rasterio Affine-like objects expose .a .b .c .d .e .f
        try:
            return (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
        except Exception:
            pass
        # fall back to iterable -> tuple and truncate/pad
        try:
            t = tuple(transform)
            if len(t) >= 6:
                return t[:6]
        except Exception:
            pass
        return None

    affine = _affine_to_tuple(depth_transform)
    if affine is not None:
        a, b, c, d, e, f = affine
        cols = np.arange(ncols, dtype=float)
        rows = np.arange(nrows, dtype=float)
        col_indices, row_indices = np.meshgrid(cols, rows)
        x_coords = a * col_indices + b * row_indices + c
        y_coords = d * col_indices + e * row_indices + f
        # set sim transforms explicitly
        sim.depth_rast_transform = (a, b, c, d, e, f)
        sim.vel_mag_rast_transform = (a, b, c, d, e, f)
        sim.vel_dir_rast_transform = (a, b, c, d, e, f)
        sim.refugia_map_transform = (a, b, c, d, e, f)
    else:
        x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
        y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
        sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        sim.vel_mag_rast_transform = sim.depth_rast_transform
        sim.vel_dir_rast_transform = sim.depth_rast_transform
        sim.refugia_map_transform = sim.depth_rast_transform

    hdf5_io.write_dataset(sim.db, 'environment/x_coords', x_coords)
    hdf5_io.write_dataset(sim.db, 'environment/y_coords', y_coords)

    # Debug: print transform types and a quick geo->pixel mapping for agents
    try:
        from emergent.salmon_abm.utils import geo_to_pixel
        print('depth_rast_transform type:', type(sim.depth_rast_transform), 'value:', sim.depth_rast_transform)
        rows, cols = geo_to_pixel(sim.X, sim.Y, sim.depth_rast_transform)
        rows = rows if hasattr(rows, '__len__') else [rows]
        cols = cols if hasattr(cols, '__len__') else [cols]
        print('geo_to_pixel rows (first 10):', np.array(rows)[:10])
        print('geo_to_pixel cols (first 10):', np.array(cols)[:10])
        # direct HDF5 read to compare
        depth_ds = hdf5_io.read_dataset(sim.db, 'environment/depth')
        valid = (np.array(rows) >= 0) & (np.array(cols) >= 0) & (np.array(rows) < depth_ds.shape[0]) & (np.array(cols) < depth_ds.shape[1])
        print('direct sampling valid count:', int(np.sum(valid)))
        if np.any(valid):
            vals = depth_ds[np.array(rows)[valid], np.array(cols)[valid]]
            print('direct sample depth (first 10 valid):', vals[:10])
    except Exception as e:
        print('Debug geo_to_pixel failed:', e)

# Prepare per-timestep agent_data datasets so we persist a time-series for later analysis
# We'll create datasets with shape (n_agents, nsteps+1) and store the initial state at column 0
n_agents = getattr(sim, 'X', None).shape[0]
nsteps = args.nsteps
if 'agent_data' not in sim.db:
    sim.db.create_group('agent_data')

# helper to create a timeseries dataset if it doesn't exist
def _create_ts(name, init_array, dtype=None):
    key = 'agent_data/' + name
    shape = (n_agents, nsteps + 1)
    if dtype is None:
        dtype = init_array.dtype
    # only create if missing
    if key not in sim.db:
        try:
            sim.db.create_dataset(key, shape=shape, dtype=dtype)
        except Exception:
            # fallback to writing with numpy array
            sim.db.create_dataset(key, data=np.zeros(shape, dtype=dtype))
    # write initial column
    try:
        sim.db[key][:, 0] = init_array
    except Exception:
        # try broadcasting
        sim.db[key][:, 0] = np.array(init_array)

# Create core time-series datasets from available sim fields or top-level datasets
_create_ts('X', getattr(sim, 'X'))
_create_ts('Y', getattr(sim, 'Y'))
_create_ts('prev_X', getattr(sim, 'prev_X'))
_create_ts('prev_Y', getattr(sim, 'prev_Y'))
# velocities if present on sim
if hasattr(sim, 'x_vel') and hasattr(sim, 'y_vel'):
    _create_ts('x_vel', getattr(sim, 'x_vel'))
    _create_ts('y_vel', getattr(sim, 'y_vel'))
# Hz (tailbeat freq)
if hasattr(sim, 'Hz'):
    _create_ts('Hz', getattr(sim, 'Hz'))
# swim behavior and stuck flags
if hasattr(sim, 'swim_behav'):
    _create_ts('swim_behav', getattr(sim, 'swim_behav'))
if hasattr(sim, 'is_stuck'):
    _create_ts('is_stuck', getattr(sim, 'is_stuck'))
# length/weight/sex
if hasattr(sim, 'length'):
    try:
        _create_ts('length', getattr(sim, 'length'))
    except Exception:
        pass
if hasattr(sim, 'weight'):
    try:
        _create_ts('weight', getattr(sim, 'weight'))
    except Exception:
        pass

# optional scalar summaries (thrust and drag magnitudes)
try:
    thrust_mag = np.linalg.norm(getattr(sim, 'thrust'), axis=1)
    _create_ts('thrust_mag', thrust_mag)
except Exception:
    pass
try:
    drag_mag = np.linalg.norm(getattr(sim, 'drag'), axis=1)
    _create_ts('drag_mag', drag_mag)
except Exception:
    pass

# run for requested timesteps and print diagnostics per step, write per-step columns
for i in range(args.nsteps):
    sim.timestep(i, 1.0)
    # Persist current state into agent_data/* datasets at column i+1
    col = i + 1
    try:
        sim.db['agent_data/X'][:, col] = sim.X
        sim.db['agent_data/Y'][:, col] = sim.Y
    except Exception:
        pass
    try:
        if 'x_vel' in sim.db and hasattr(sim, 'x_vel'):
            sim.db['agent_data/x_vel'][:, col] = sim.x_vel
        if 'y_vel' in sim.db and hasattr(sim, 'y_vel'):
            sim.db['agent_data/y_vel'][:, col] = sim.y_vel
    except Exception:
        pass
    try:
        if 'Hz' in sim.db and hasattr(sim, 'Hz'):
            sim.db['agent_data/Hz'][:, col] = sim.Hz
    except Exception:
        pass
    try:
        if 'swim_behav' in sim.db and hasattr(sim, 'swim_behav'):
            sim.db['agent_data/swim_behav'][:, col] = sim.swim_behav
    except Exception:
        pass
    try:
        if 'is_stuck' in sim.db and hasattr(sim, 'is_stuck'):
            sim.db['agent_data/is_stuck'][:, col] = sim.is_stuck
    except Exception:
        pass
    try:
        if 'thrust_mag' in sim.db and hasattr(sim, 'thrust'):
            tmag = np.linalg.norm(sim.thrust, axis=1)
            sim.db['agent_data/thrust_mag'][:, col] = tmag
    except Exception:
        pass
    try:
        if 'drag_mag' in sim.db and hasattr(sim, 'drag'):
            dmag = np.linalg.norm(sim.drag, axis=1)
            sim.db['agent_data/drag_mag'][:, col] = dmag
    except Exception:
        pass

    mean_Hz = np.nanmean(sim.Hz) if hasattr(sim, 'Hz') else float('nan')
    mean_thrust = np.nanmean(np.linalg.norm(sim.thrust, axis=1)) if hasattr(sim, 'thrust') else float('nan')
    mean_drag = np.nanmean(np.linalg.norm(sim.drag, axis=1)) if hasattr(sim, 'drag') else float('nan')
    mean_speed = np.nanmean(np.linalg.norm(np.stack((getattr(sim,'x_vel', np.zeros_like(sim.X)), getattr(sim,'y_vel', np.zeros_like(sim.Y))), axis=-1), axis=1))
    moved = np.sum((sim.X != sim.prev_X) | (sim.Y != sim.prev_Y))
    print(f'step {i}: mean_Hz={mean_Hz:.3f}, mean_thrust={mean_thrust:.3e}, mean_drag={mean_drag:.3e}, mean_speed={mean_speed:.3f}, moved_agents={moved}')

# sample a few environment values for agents 0..4
print('sample depth for agents 0..4:', sim.sample_environment(sim.depth_rast_transform, 'depth')[:5])
print('sample vel_x for agents 0..4:', sim.sample_environment(sim.vel_dir_rast_transform, 'vel_x')[:5])

# End-of-run diagnostics: print small slices of key arrays and persist freq_debug_history
try:
    print('\nEnd-of-run diagnostics:')
    print('Hz[:20]:', getattr(sim, 'Hz', None)[:20])
    print('thrust[:20]:', getattr(sim, 'thrust', None)[:20])
    print('drag[:20]:', getattr(sim, 'drag', None)[:20])
    print('X[:20]:', getattr(sim, 'X', None)[:20])

    # freq_debug snapshot if present
    print('freq_debug (most recent):', getattr(sim, 'freq_debug', None))
    fd_hist = getattr(sim, 'freq_debug_history', None)
    print('freq_debug_history length:', 0 if fd_hist is None else len(fd_hist))

    import json

    # convert numpy arrays to native Python types for JSON serialization
    def _make_serializable(o):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if isinstance(o, dict):
            return {k: _make_serializable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_make_serializable(v) for v in o]
        if _np is not None and isinstance(o, _np.ndarray):
            return _make_serializable(o.tolist())
        # numpy scalar types
        if _np is not None and isinstance(o, (_np.integer, _np.floating)):
            return o.item()
        return o

    # persist freq_debug_history if it exists
    if fd_hist is not None and len(fd_hist) > 0:
        try:
            serial = [_make_serializable(d) for d in fd_hist]
            hdf5_io.write_dataset(sim.db, 'diagnostics/freq_debug_history_json', np.array([json.dumps(serial)]))
            print('Wrote diagnostics/freq_debug_history_json to DB')
        except Exception as e:
            print('Failed to write freq_debug_history to DB:', e)

    # persist the term-level history if present
    fth = getattr(sim, 'freq_terms_history', None)
    if fth is not None and len(fth) > 0:
        try:
            serial_terms = _make_serializable(fth)
            hdf5_io.write_dataset(sim.db, 'diagnostics/freq_terms_history_json', np.array([json.dumps(serial_terms)]))
            print('Wrote diagnostics/freq_terms_history_json to DB')
        except Exception as e:
            print('Failed to write freq_terms_history to DB:', e)
        # also write a JSON copy into outputs/ for easier inspection
        try:
            out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
            out_dir = os.path.abspath(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            import time
            fname = f"diagnostics_freq_terms_{int(time.time())}.json"
            out_path = os.path.join(out_dir, fname)
            with open(out_path, 'w') as of:
                json.dump(serial_terms, of)
            print('Wrote JSON diagnostics copy to', out_path)
        except Exception as e:
            print('Failed to write JSON diagnostics copy:', e)
except Exception as e:
    print('Error printing end-of-run diagnostics:', e)

# Close DB after diagnostics persistence
try:
    sim.db.close()
    print('DB path:', sim.db_path)
except Exception:
    pass
