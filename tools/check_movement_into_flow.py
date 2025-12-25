import os
import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io

# small diagnostic: 10 agents, 1 timestep
model_dir = 'outputs'
model_name = 'check_flow'
crs = None
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
env_dir = os.path.join('data', 'salmon_abm')
env_files = []
for fn in ('vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif'):
    p = os.path.join(env_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=2, num_agents=10)

# ensure rasters are in DB: import env_files into HDF5 so sampling picks them up
h5 = getattr(sim, 'db', None)
vx = vy = None
if h5 is not None:
    from emergent.salmon_abm import io as _io
    for ef in env_files:
        try:
            arr, transform, crs = _io.enviro_import(ef)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            hdf5_io.write_dataset(h5, key, arr)
        except Exception as e:
            print('env import->db failed for', ef, e)
    vx = hdf5_io.read_dataset(h5, 'environment/vel_x')
    vy = hdf5_io.read_dataset(h5, 'environment/vel_y')
    # write x/y coords if missing
    try:
        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth')
        if depth_ds is not None:
            nrows, ncols = np.array(depth_ds).shape
            cols = np.arange(ncols, dtype=float)
            rows_idx = np.arange(nrows, dtype=float)
            col_indices, row_indices = np.meshgrid(cols, rows_idx)
            x_coords = col_indices
            y_coords = row_indices
            hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
            hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
    except Exception:
        pass
    # reinitialize headings from DB after writing rasters
    try:
        sim.initialize_headings_from_db()
    except Exception:
        pass

# snapshot water velocities sampled into the sim by initialize_headings_from_db
water_vx = getattr(sim, 'x_vel', None)
water_vy = getattr(sim, 'y_vel', None)
if water_vx is None or water_vy is None:
    water_vx = np.zeros(sim.num_agents)
    water_vy = np.zeros(sim.num_agents)

# run one timestep
sim.timestep(0, 1.0)

# compute displacement vectors
disp_x = sim.X - sim.prev_X
disp_y = sim.Y - sim.prev_Y

# compute alignment with -water velocity (into flow)
results = []
for i in range(sim.num_agents):
    w = np.array([-water_vx[i], -water_vy[i]])
    d = np.array([disp_x[i], disp_y[i]])
    w_norm = np.linalg.norm(w)
    d_norm = np.linalg.norm(d)
    if w_norm == 0 or d_norm == 0:
        cos = np.nan
    else:
        cos = np.dot(w, d) / (w_norm * d_norm)
    results.append((i, cos, w_norm, d_norm))

# report
cos_vals = np.array([r[1] for r in results], dtype=float)
valid_cos = np.isfinite(cos_vals)
print('Per-agent cosine with -water_velocity (NaN means zero water or zero movement):')
for r in results:
    print(f'agent {r[0]:3d}: cos={r[1]:6.3f}, |water|={r[2]:6.3f}, |disp|={r[3]:6.3f}')

if np.any(valid_cos):
    mean_cos = np.nanmean(cos_vals)
    frac_good = np.sum(cos_vals > 0.5) / np.sum(valid_cos)
    print('\nMean cos (agents with valid cos):', mean_cos)
    print('Fraction with cos > 0.5 (roughly into-flow):', frac_good)
else:
    print('\nNo valid cos values to report')

# cleanup
sim.close()
