"""Run a headless simulation: 1000 agents × 600 timesteps and persist trajectories.

Outputs:
 - HDF5 DB in outputs/diagnostics_start1000_run/sim_db_start1000_run.h5
 - trajectories NPZ: outputs/diagnostics_start1000_run/trajectories_1000x600.npz
"""
import os, time
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import numpy as np

base = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base, 'data', 'salmon_abm')
start_poly = os.path.join(data_dir, 'start_loc_river_right.shp')
env_files = [os.path.join(data_dir, fname) for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']]

outdir = os.path.abspath(os.path.join(base, 'outputs', 'diagnostics_start1000_run'))
os.makedirs(outdir, exist_ok=True)

n_agents = 1000
n_steps = 600

print(f'Running headless sim: {n_agents} agents for {n_steps} timesteps; outputs -> {outdir}')

sim = simulation(model_dir=outdir, model_name='start1000_run', crs=None, basin='Nushagak River', water_temp=10.0, start_polygon=start_poly, env_files=env_files, longitudinal_profile=None, num_timesteps=n_steps, num_agents=n_agents)
# Import environment rasters into DB if not present
h5obj = getattr(sim, 'db', None)
if h5obj is not None:
    for ef in env_files:
        try:
            arr, transform, crs = io.enviro_import(ef)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            hdf5_io.write_dataset(h5obj, key, arr)
        except Exception as e:
            print('env import failed for', ef, e)
    # write x/y coords
    depth_ds = hdf5_io.read_dataset(h5obj, 'environment/depth')
    if depth_ds is not None:
        nrows, ncols = depth_ds.shape
        try:
            _, depth_transform, _ = io.enviro_import(env_files[0])
            a,b,c,d,e,f = depth_transform.a, depth_transform.b, depth_transform.c, depth_transform.d, depth_transform.e, depth_transform.f
            cols = np.arange(ncols, dtype=float)
            rows_idx = np.arange(nrows, dtype=float)
            col_indices, row_indices = np.meshgrid(cols, rows_idx)
            x_coords = a * col_indices + b * row_indices + c
            y_coords = d * col_indices + e * row_indices + f
            sim.depth_rast_transform = (a,b,c,d,e,f)
            sim.vel_mag_rast_transform = (a,b,c,d,e,f)
            sim.vel_dir_rast_transform = (a,b,c,d,e,f)
            sim.refugia_map_transform = (a,b,c,d,e,f)
        except Exception:
            x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
            y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
            sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            sim.vel_mag_rast_transform = sim.depth_rast_transform
            sim.vel_dir_rast_transform = sim.depth_rast_transform
            sim.refugia_map_transform = sim.depth_rast_transform
        hdf5_io.write_dataset(h5obj, 'environment/x_coords', x_coords)
        hdf5_io.write_dataset(h5obj, 'environment/y_coords', y_coords)

# reinitialize headings from DB
try:
    sim.initialize_headings_from_db()
except Exception:
    pass

# Prepare arrays to store trajectories: shape (n_steps+1, n_agents)
traj_X = np.full((n_steps+1, n_agents), np.nan, dtype=np.float64)
traj_Y = np.full((n_steps+1, n_agents), np.nan, dtype=np.float64)
traj_X[0, :] = sim.X
traj_Y[0, :] = sim.Y

start_time = time.time()
for t in range(n_steps):
    sim.timestep(t, 1.0)
    traj_X[t+1, :] = sim.X
    traj_Y[t+1, :] = sim.Y
    if (t+1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f'step {t+1}/{n_steps} elapsed {elapsed:.1f}s')

# Save trajectories NPZ and flush DB
traj_fname = os.path.join(outdir, f'trajectories_{n_agents}x{n_steps}.npz')
np.savez_compressed(traj_fname, X=traj_X, Y=traj_Y)
print('Wrote trajectories to', traj_fname)

# Also ensure DB is flushed and closed
try:
    if hasattr(sim.db, 'flush'):
        sim.db.flush()
    sim.db.close()
except Exception:
    pass

print('Done')
