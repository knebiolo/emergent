"""Full HECRAS smoke test: map nodes->env rasters, build distance raster, extract centerline."""
import os, sys
if len(sys.argv) < 2:
    print('Usage: python hecras_centerline_full_smoketest.py <hdf_path>')
    sys.exit(2)

hdf_path = sys.argv[1]
if not os.path.exists(hdf_path):
    print('HDF not found:', hdf_path); sys.exit(2)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import (
    ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters,
    derive_centerline_from_distance_raster
)

class DummySim:
    def __init__(self, hdf_path):
        # open in r+ so mapping functions can create datasets
        self.hdf5 = h5py.File(hdf_path, 'r+')
        self.num_agents = 10
        self.flush_interval = 1
        self.num_timesteps = 1
        self._hdf5_buffers = {}
        self._buffer_pos = 0
        self.hecras_plan_path = hdf_path
        self.hecras_fields = ['Cell Hydraulic Depth', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y']
        self._hecras_maps = {}

    def close(self):
        try:
            self.hdf5.close()
        except Exception:
            pass

sim = DummySim(hdf_path)
try:
    print('Ensuring hdf coords from HECRAS...')
    ensure_hdf_coords_from_hecras(sim, hdf_path)
    print('depth_rast_transform:', getattr(sim, 'depth_rast_transform', None))

    print('Mapping HECRAS node fields to environment rasters...')
    ok = map_hecras_to_env_rasters(sim, hdf_path, field_names=sim.hecras_fields, k=8)
    print('map_hecras_to_env_rasters returned', ok)

    env = sim.hdf5.get('environment')
    if env is None:
        print('No environment group after mapping; abort')
        sys.exit(1)

    # prefer existing distance_to
    if 'distance_to' in env:
        distance = np.array(env['distance_to'])
        print('Found distance_to in environment; shape', distance.shape)
    else:
        # try wetted or depth
        distance = None
        if 'wetted' in env:
            wetted = np.array(env['wetted'])
            mask = (wetted != -9999) & (wetted > 0)
            try:
                from scipy.ndimage import distance_transform_edt
                pixel_width = getattr(sim, 'depth_rast_transform', None).a if getattr(sim, 'depth_rast_transform', None) is not None else 1.0
                distance = distance_transform_edt(mask) * abs(pixel_width)
                print('Computed distance from wetted; shape', distance.shape)
            except Exception as e:
                print('Failed to compute distance from wetted:', e)
        elif 'depth' in env:
            depth = np.array(env['depth'])
            mask = np.isfinite(depth) & (depth > 0)
            try:
                from scipy.ndimage import distance_transform_edt
                pixel_width = getattr(sim, 'depth_rast_transform', None).a if getattr(sim, 'depth_rast_transform', None) is not None else 1.0
                distance = distance_transform_edt(mask) * abs(pixel_width)
                print('Computed distance from depth; shape', distance.shape)
            except Exception as e:
                print('Failed to compute distance from depth:', e)

    if distance is None:
        print('No distance raster available after mapping; abort')
        sys.exit(1)

    print('Running centerline extraction helper...')
    main_centerline, all_lines = derive_centerline_from_distance_raster(distance, transform=getattr(sim, 'depth_rast_transform', None), footprint_size=5, min_length=10)
    
    # Print results matching visualization script format
    print(f'Raster shape: {distance.shape}')
    print(f'Distance range: {distance.min():.2f} to {distance.max():.2f} m')
    from scipy.ndimage import maximum_filter
    from skimage.morphology import skeletonize
    local_max = maximum_filter(distance, size=5)
    is_ridge = (distance == local_max) & (distance > 0.5)
    print(f'Ridge pixels: {is_ridge.sum()}')
    
    for i, line in enumerate(all_lines, 1):
        print(f'  Path {i}: {len(line.coords)} pts, {line.length:.1f}m')
    
    print(f'\nTotal centerlines: {len(all_lines)}')
    if main_centerline:
        print(f'Main centerline: {main_centerline.length:.1f}m')

finally:
    sim.close()

print('Done')
