import os
import numpy as np
import matplotlib.pyplot as plt
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io as _io, hdf5_io

model_dir = 'outputs'
model_name = 'snapshot'
crs = None
basin = 'Nushagak'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
env_dir = os.path.join('data', 'salmon_abm')
env_files = [os.path.join(env_dir, fn) for fn in ('depth.tif','vel_x.tif','vel_y.tif','vel_mag.tif','vel_dir.tif') if os.path.exists(os.path.join(env_dir, fn))]

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=10, num_agents=50)

# ensure rasters in DB
h5 = getattr(sim, 'db', None)
for ef in env_files:
    try:
        arr, tr, crs = _io.write_raster_to_hdf5(h5, ef, sim=sim)
    except Exception:
        pass

# run a few timesteps
for t in range(5):
    sim.timestep(t, 1.0)

# read depth array from hdf5 for background
depth = hdf5_io.read_dataset(h5, 'environment/depth', default=None)
if depth is None:
    # fallback: try read via io
    try:
        depth, tr, crs = _io.enviro_import(os.path.join(env_dir,'depth.tif'))
    except Exception:
        depth = np.zeros((200,200))

# simple plotting
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(depth, cmap='viridis', origin='upper')
ax.scatter((sim.X - sim.X.min()) / (sim.X.max()-sim.X.min()) * depth.shape[1],
           (sim.Y - sim.Y.min()) / (sim.Y.max()-sim.Y.min()) * depth.shape[0],
           c='red', s=10)
ax.set_title('Headless snapshot (agents overlay)')
out = os.path.join('outputs','snapshot_agents.png')
plt.savefig(out, dpi=150)
print('wrote', out)
sim.close()
