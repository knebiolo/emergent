import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))

import numpy as np
from emergent.salmon_abm.sockeye_SoA import simulation
from shapely.geometry import LineString, Polygon

# Monkeypatch geopandas.read_file like the profiler harness to avoid needing a shapefile
try:
    import geopandas as gpd
    _orig_read = gpd.read_file
    def _dummy_read_file(*args, **kwargs):
        return gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(9,0),(9,9),(0,9)])]}, crs='EPSG:4326')
    gpd.read_file = _dummy_read_file
except Exception:
    gpd = None

# Monkeypatch enviro_import to create synthetic rasters like profiler
from rasterio.transform import Affine
def enviro_import_dummy(self, data_dir, surface_type):
    height = 64
    width = 64
    transform = Affine.translation(0, 0) * Affine.scale(1, -1)
    if 'environment' not in self.hdf5:
        env_data = self.hdf5.create_group('environment')
    else:
        env_data = self.hdf5['environment']
    arr = np.zeros((height, width), dtype='float32')
    keymap = {'wetted':'wetted','velocity x':'vel_x','velocity y':'vel_y','depth':'depth','wsel':'wsel','elevation':'elevation','velocity direction':'vel_dir','velocity magnitude':'vel_mag'}
    key = keymap.get(surface_type, surface_type)
    if key not in env_data:
        env_data.create_dataset(key, (height, width), dtype='f4')
    env_data[key][:,:] = arr
    self.width = width
    self.height = height
    self.depth_rast_transform = transform
    self.vel_x_rast_transform = transform
    self.vel_y_rast_transform = transform
    self.vel_dir_rast_transform = transform
    self.vel_mag_rast_transform = transform
    self.wsel_rast_transform = transform
    self.elev_rast_transform = transform
    self.wetted_transform = transform

from emergent.salmon_abm.sockeye_SoA import simulation as _simcls
_simcls.enviro_import = enviro_import_dummy

# Create a tiny simulation using the existing shallow enviro_import monkeypatch
env_files = {k: '' for k in ['x_vel','y_vel','depth','wsel','elev','vel_dir','vel_mag','wetted']}

sim = simulation(model_dir=str(root), model_name='hec_test', crs='EPSG:4326', basin='Nushagak River',
                  water_temp=10.0, start_polygon=None, env_files=env_files,
                 longitudinal_profile=None, fish_length=500, num_timesteps=10, num_agents=10, use_gpu=False)

# Place agents in a grid for test
sim.X = np.linspace(0, 9, sim.num_agents).astype(float)
sim.Y = np.linspace(0, 9, sim.num_agents).astype(float)

# Build synthetic irregular HECRAS nodes
# e.g., 30 random nodes in domain [0,9]x[0,9]
rng = np.random.RandomState(0)
nodes = rng.rand(30, 2) * 9.0

# Create nodal fields: depth increases with x, vel_x random, vel_y random
depth_nodes = nodes[:, 0] * 0.1 + 0.5
vel_x_nodes = (rng.rand(30) - 0.5) * 0.2
vel_y_nodes = (rng.rand(30) - 0.5) * 0.2

# Build mapping from nodes to agent points
agent_points = np.vstack([sim.X.flatten(), sim.Y.flatten()]).T
sim.build_hecras_mapping(nodes, agent_points, k=4)

# Attach nodal fields
sim.hecras_node_fields = {'depth': depth_nodes, 'vel_x': vel_x_nodes, 'vel_y': vel_y_nodes}

# Apply mapping and print results
depth_agents = sim.apply_hecras_mapping(depth_nodes)
velx_agents = sim.apply_hecras_mapping(vel_x_nodes)
vely_agents = sim.apply_hecras_mapping(vel_y_nodes)

print('Agent depths:', depth_agents)
print('Agent vel_x:', velx_agents)
print('Agent vel_y:', vely_agents)
