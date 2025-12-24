import os
from emergent.salmon_abm.simulation import simulation

# small diagnostic: 5 agents, 1 timestep
model_dir = 'outputs'
model_name = 'diag_start_velocity'
crs = None
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
# include raster env files if present
env_dir = os.path.join('data', 'salmon_abm')
env_files = []
for fn in ('vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif'):
    p = os.path.join(env_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

print('Using env_files:', env_files)

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=2, num_agents=5)
# enable debugging to capture movement/frequency internals
sim.debug_freq = True
sim.debug_movement = True

print('\nInitial state:')
print('X[:5]=', sim.X[:5])
print('Y[:5]=', sim.Y[:5])
print('heading[:5]=', sim.heading[:5])
print('sog[:5]=', sim.sog[:5])
print('ideal_sog[:5]=', sim.ideal_sog[:5])
print('x_vel[:5]=', sim.x_vel[:5])
print('y_vel[:5]=', sim.y_vel[:5])

# run one timestep
ok = sim.timestep(0, 1.0)
print('\nAfter one timestep:')
print('X[:5]=', sim.X[:5])
print('Y[:5]=', sim.Y[:5])
print('x_vel[:5]=', sim.x_vel[:5])
print('y_vel[:5]=', sim.y_vel[:5])

# Also print displacement magnitudes
import numpy as np
disp = np.sqrt((sim.X - sim.prev_X)**2 + (sim.Y - sim.prev_Y)**2)
print('displacements[:5]=', disp[:5])

# print whether any moved
print('any moved?', np.any(disp > 0))

# cleanup
sim.close()
