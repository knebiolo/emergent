import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import utils
from scipy.spatial.distance import pdist, squareform
import os

sim = simulation(model_dir='outputs/diagnostics', model_name='check_init', crs=None, basin=None, water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None, num_timesteps=1, num_agents=4)
print('num_agents', sim.num_agents)
print('X initial:', sim.X)
print('Y initial:', sim.Y)
coords = np.column_stack((sim.X, sim.Y))
print('pairwise distances:')
print(squareform(pdist(coords)))
# Pixel mapping
rows, cols = utils.geo_to_pixel(sim.X, sim.Y, sim.depth_rast_transform)
print('rows:', rows)
print('cols:', cols)
# Check whether all agents map to same (row,col)
pairs = list(zip(rows, cols))
print('agent pixel pairs:', pairs)
all_same = all(p == pairs[0] for p in pairs)
print('All agents in same pixel cell?', all_same)
# Print heading init
print('initial heading angles:', sim.heading)
# Distances to each agent from agent 0
dists0 = np.sqrt((sim.X - sim.X[0])**2 + (sim.Y - sim.Y[0])**2)
print('distances from agent 0:', dists0)

# If environment grids exist, print grid cell centers of those pixels
try:
    h5 = sim.db
    if 'environment/x_coords' in h5 and 'environment/y_coords' in h5:
        x_coords = h5['environment/x_coords'][:]
        y_coords = h5['environment/y_coords'][:]
        centers = [ (float(x_coords[r,c]), float(y_coords[r,c])) for r,c in pairs ]
        print('pixel centers for agents:', centers)
except Exception as e:
    print('could not read x_coords/y_coords from db:', e)
