"""Run a 100-agent simulation for 10 timesteps using real rasters from data/salmon_abm."""
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Run instrumented sim with real env rasters')
parser.add_argument('--nsteps', type=int, default=10, help='number of timesteps to run')
parser.add_argument('--nagents', type=int, default=100, help='number of agents to simulate')
parser.add_argument('--outdir', type=str, default=None, help='output directory to save DB (overrides model_dir)')
parser.add_argument('--model_name', type=str, default='real_probe', help='model name used for the HDF5 filename')
parser.add_argument('--debug', action='store_true', help='enable verbose debug output and diagnostic JSON writes')
args = parser.parse_args()

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
# Build env files list: depth, vel_x, vel_y, vel_mag, vel_dir
env_files = [os.path.join(data_dir, fname) for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']]
start_poly = os.path.join(data_dir, 'start_loc_river_right.shp')

# determine model_dir / outputs path
outdir = args.outdir or os.path.join(os.path.dirname(__file__), '..', 'outputs')
outdir = os.path.abspath(outdir)
os.makedirs(outdir, exist_ok=True)

sim = simulation(model_dir=outdir, model_name=args.model_name, crs=None, basin='test', water_temp=10, start_polygon=start_poly, env_files=env_files, longitudinal_profile=None, fish_length=200.0, num_timesteps=args.nsteps, num_agents=args.nagents)

# enable debug flags for diagnostics when requested
if args.debug:
    sim.debug_env = True
    sim.debug_freq = True
    # enable fine-grained frequency probe CSV output
    sim.debug_freq_probe = True
else:
    sim.debug_env = False
    sim.debug_freq = False
    sim.debug_freq_probe = False

# Use the HDF5 object returned by hdf5_io.get_hdf5_obj(sim) if available,
# otherwise fall back to the simulation's hdf5 attribute.
h5obj = hdf5_io.get_hdf5_obj(sim) or getattr(sim, 'hdf5', None)

# Import rasters into the HDF5 object using io.enviro_import if available
for ef in env_files:
    try:
        arr, transform, crs = io.enviro_import(ef)
        # write arrays in environment/ prefix keyed by base filename without extension
        key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
        hdf5_io.write_dataset(h5obj, key, arr)
    except Exception as e:
        print('env import failed for', ef, e)

# write x/y coords and transforms consistent with rasters
# attempt to read back one raster to get shape
depth = hdf5_io.read_dataset(h5obj, 'environment/depth')
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

    hdf5_io.write_dataset(h5obj, 'environment/x_coords', x_coords)
    hdf5_io.write_dataset(h5obj, 'environment/y_coords', y_coords)

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

# Prepare agent datasets using centralized helpers for cleanliness
nsteps = args.nsteps
if h5obj is None:
    raise RuntimeError('No HDF5 object available for writing')

# create agent time-series and scalar datasets (no-op if already present)
hdf5_io.create_agent_timeseries(h5obj, sim, nsteps)

# run for requested timesteps and print diagnostics per step, write per-step columns
for i in range(args.nsteps):
    sim.timestep(i, 1.0)
    # Persist current state into agent_data/* at column i via helper
    hdf5_io.write_agent_timestep(h5obj, sim, i)

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
    if getattr(sim, 'debug_freq', False):
        if fd_hist is not None and len(fd_hist) > 0:
            try:
                serial = [_make_serializable(d) for d in fd_hist]
                hdf5_io.write_dataset(sim.db, 'diagnostics/freq_debug_history_json', np.array([json.dumps(serial)]))
                print('Wrote diagnostics/freq_debug_history_json to DB')
            except Exception as e:
                print('Failed to write freq_debug_history to DB:', e)

    # persist the term-level history if present
    fth = getattr(sim, 'freq_terms_history', None)
    if getattr(sim, 'debug_freq', False):
        if fth is not None and len(fth) > 0:
            try:
                serial_terms = _make_serializable(fth)
                hdf5_io.write_dataset(sim.db, 'diagnostics/freq_terms_history_json', np.array([json.dumps(serial_terms)]))
                print('Wrote diagnostics/freq_terms_history_json to DB')
            except Exception as e:
                print('Failed to write freq_terms_history to DB:', e)
            # also write a JSON copy into outputs/ for easier inspection when debug
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
