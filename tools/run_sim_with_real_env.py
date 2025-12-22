"""Run a 100-agent simulation for 10 timesteps using real rasters from data/salmon_abm."""
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import os
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
# Build env files list: depth, vel_x, vel_y, vel_mag, vel_dir
env_files = [os.path.join(data_dir, fname) for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']]
start_poly = os.path.join(data_dir, 'start_loc_river_right.shp')

sim = simulation(model_dir='.', model_name='real_probe', crs=None, basin='test', water_temp=10, start_polygon=start_poly, env_files=env_files, longitudinal_profile=None, fish_length=200.0, num_timesteps=10, num_agents=100)

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

    if depth_transform is not None and not callable(depth_transform):
        a, b, c, d, e, f = depth_transform
        cols = np.arange(ncols, dtype=float)
        rows = np.arange(nrows, dtype=float)
        col_indices, row_indices = np.meshgrid(cols, rows)
        x_coords = a * col_indices + b * row_indices + c
        y_coords = d * col_indices + e * row_indices + f
        # set sim transforms explicitly
        sim.depth_rast_transform = depth_transform
        sim.vel_mag_rast_transform = depth_transform
        sim.vel_dir_rast_transform = depth_transform
        sim.refugia_map_transform = depth_transform
    else:
        x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
        y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
        sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        sim.vel_mag_rast_transform = sim.depth_rast_transform
        sim.vel_dir_rast_transform = sim.depth_rast_transform
        sim.refugia_map_transform = sim.depth_rast_transform

    hdf5_io.write_dataset(sim.db, 'environment/x_coords', x_coords)
    hdf5_io.write_dataset(sim.db, 'environment/y_coords', y_coords)

# run for 10 timesteps and print diagnostics per step
for i in range(10):
    sim.timestep(i, 1.0)
    mean_Hz = np.nanmean(sim.Hz)
    mean_thrust = np.nanmean(np.linalg.norm(sim.thrust, axis=1))
    mean_drag = np.nanmean(np.linalg.norm(sim.drag, axis=1))
    mean_speed = np.nanmean(np.linalg.norm(np.stack((sim.x_vel, sim.y_vel), axis=-1), axis=1))
    moved = np.sum((sim.X != sim.prev_X) | (sim.Y != sim.prev_Y))
    print(f'step {i}: mean_Hz={mean_Hz:.3f}, mean_thrust={mean_thrust:.3e}, mean_drag={mean_drag:.3e}, mean_speed={mean_speed:.3f}, moved_agents={moved}')

# sample a few environment values for agents 0..4
print('sample depth for agents 0..4:', sim.sample_environment(sim.depth_rast_transform, 'depth')[:5])
print('sample vel_x for agents 0..4:', sim.sample_environment(sim.vel_dir_rast_transform, 'vel_x')[:5])

sim.db.close()
print('DB path:', sim.db_path)
