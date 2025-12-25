import os
import numpy as np
from emergent.salmon_abm.simulation import simulation

# small diagnostic: 10 agents, 1 timestep
model_dir = 'outputs'
model_name = 'check_flow2'
crs = None
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
env_dir = os.path.join('data', 'salmon_abm')
env_files = []
for fn in ('depth.tif','vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif'):
    p = os.path.join(env_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=2, num_agents=10)

# ensure rasters are available (write env files into DB)
from emergent.salmon_abm import io, hdf5_io
h5 = getattr(sim, 'db', None)
if h5 is not None:
    for ef in env_files:
        try:
            arr, transform, crs = io.enviro_import(ef)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            hdf5_io.write_dataset(h5, key, arr)
        except Exception as e:
            print('env import->db failed for', ef, e)
    try:
        sim.initialize_headings_from_db()
    except Exception:
        pass

# sample environment via provided helper
water_vx = sim.sample_environment(sim.depth_rast_transform, 'vel_x')
water_vy = sim.sample_environment(sim.depth_rast_transform, 'vel_y')

# run one timestep
sim.timestep(0,1.0)

# compute displacement vectors
disp_x = sim.X - sim.prev_X
disp_y = sim.Y - sim.prev_Y

# compute alignment
cos_vals = []
for i in range(sim.num_agents):
    w = np.array([-water_vx[i], -water_vy[i]])
    d = np.array([disp_x[i], disp_y[i]])
    if np.linalg.norm(w) == 0 or np.linalg.norm(d) == 0:
        cos = np.nan
    else:
        cos = np.dot(w, d) / (np.linalg.norm(w) * np.linalg.norm(d))
    cos_vals.append(cos)

cos_vals = np.array(cos_vals)
valid = np.isfinite(cos_vals)
print('mean cos (valid):', np.nanmean(cos_vals))
print('frac cos>0.5:', np.sum(cos_vals>0.5)/np.sum(valid) if np.sum(valid)>0 else None)
print('sample cos values:', cos_vals[:10])

sim.close()
