import sys
import os
from pathlib import Path
import cProfile
import pstats

# Ensure src is on sys.path
root = Path(__file__).resolve().parents[1]
src = root / 'src'
sys.path.insert(0, str(src))

# Import necessary modules
import emergent.salmon_abm.sockeye_SoA as sockeye_mod
from shapely.geometry import Polygon, LineString

# Monkeypatch geopandas.read_file to avoid reading from network files
try:
    import geopandas as gpd
    _orig_read = gpd.read_file
    def _dummy_read_file(*args, **kwargs):
        return gpd.GeoDataFrame({'geometry':[Polygon([(0,0),(1,0),(1,1),(0,1)])]}, crs='EPSG:4326')
    gpd.read_file = _dummy_read_file
except Exception:
    gpd = None

# Monkeypatch enviro_import to create small synthetic environment arrays
from rasterio.transform import Affine
import numpy as np
import h5py

def enviro_import_dummy(self, data_dir, surface_type):
    height = 64
    width = 64
    transform = Affine.translation(0, 0) * Affine.scale(1, -1)
    if 'environment' not in self.hdf5:
        env_data = self.hdf5.create_group('environment')
    else:
        env_data = self.hdf5['environment']

    # Write coordinate grids if missing
    if 'x_coords' not in self.hdf5:
        dset_x = self.hdf5.create_dataset('x_coords', (height, width), dtype='float32')
        dset_y = self.hdf5.create_dataset('y_coords', (height, width), dtype='float32')
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        x_coords = cols.astype('float32')
        y_coords = rows.astype('float32')
        self.hdf5['x_coords'][:, :] = x_coords
        self.hdf5['y_coords'][:, :] = y_coords

    arr = np.zeros((height, width), dtype='float32')
    # create dataset names mapping for surface types
    name_map = {
        'wetted': 'wetted',
        'velocity x': 'vel_x',
        'velocity y': 'vel_y',
        'depth': 'depth',
        'wsel': 'wsel',
        'elevation': 'elevation',
        'velocity direction': 'vel_dir',
        'velocity magnitude': 'vel_mag'
    }
    key = name_map.get(surface_type, surface_type)
    if key not in env_data:
        env_data.create_dataset(key, (height, width), dtype='f4')
    env_data[key][:, :] = arr

    # set transforms and dims on object
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

# Patch the class method
sockeye_mod.simulation.enviro_import = enviro_import_dummy

# Also patch any heavy external calls if needed (e.g., rasterio.open inside module) - avoided by enviro_import patch

# Create small simulation instance
env_files = {k: '' for k in ['x_vel','y_vel','depth','wsel','elev','vel_dir','vel_mag','wetted']}

sim = sockeye_mod.simulation(model_dir=str(root), model_name='prof_min', crs='EPSG:4326', basin='Nushagak River',
                              water_temp=10.0, start_polygon=None, env_files=env_files,
                              longitudinal_profile=None, fish_length=500, num_timesteps=200, num_agents=50, use_gpu=False)

# Use a simple LineString for longitudinal profile to avoid LinearIterator errors
sim.longitudinal = LineString([(0, 0), (0, 63)])

# Create a basic PID controller
pid = sockeye_mod.PID_controller(n_agents=50, k_p=0.1, k_i=0.0, k_d=0.0)

# Profile a number of timesteps calling sim.timestep directly
pr = cProfile.Profile()
pr.enable()
try:
    for t in range(500):
        sim.timestep(t, 1.0, None, pid)
except Exception as e:
    print('timestep raised exception:', e)
pr.disable()

# Print top callers
p = pstats.Stats(pr).strip_dirs().sort_stats('cumtime')
p.print_stats(30)
print('Done')

# Restore geopandas read_file
if gpd is not None:
    gpd.read_file = _orig_read
