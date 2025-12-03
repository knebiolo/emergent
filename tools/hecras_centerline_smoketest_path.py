"""Run centerline extraction on a provided HECRAS HDF path and print results.

Usage:
  python hecras_centerline_smoketest_path.py <path-to-hdf>
"""
import sys, os
if len(sys.argv) < 2:
    print('Usage: python hecras_centerline_smoketest_path.py <path-to-hdf>')
    sys.exit(2)

hdf_path = sys.argv[1]
if not os.path.exists(hdf_path):
    print('HDF path does not exist:', hdf_path)
    sys.exit(2)

import numpy as np
import h5py
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# import helper functions from module
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import derive_centerline_from_distance_raster, compute_affine_from_hecras

print('Inspecting HDF:', hdf_path)
with h5py.File(hdf_path, 'r') as hf:
    env = hf.get('environment')
    distance = None
    wetted = None
    x_coords = None
    y_coords = None
    if env is not None:
        if 'distance_to' in env:
            try:
                distance = np.array(env['distance_to'])
                print('Found environment/distance_to with shape', distance.shape)
            except Exception as e:
                print('Failed reading environment/distance_to:', e)
        if 'wetted' in env:
            try:
                wetted = np.array(env['wetted'])
                print('Found environment/wetted with shape', wetted.shape)
            except Exception as e:
                print('Failed reading environment/wetted:', e)
        if 'x_coords' in env and 'y_coords' in env:
            try:
                x_coords = np.array(env['x_coords'])
                y_coords = np.array(env['y_coords'])
                print('Found x_coords/y_coords shapes', getattr(x_coords,'shape',None), getattr(y_coords,'shape',None))
            except Exception as e:
                print('Failed reading x_coords/y_coords:', e)
    else:
        print('No environment group in HDF')

    # fallback: see if HECRAS plan stores distance_to elsewhere
    if distance is None:
        # try common paths
        if '/environment/distance_to' in hf:
            try:
                distance = np.array(hf['/environment/distance_to'])
                print('Found /environment/distance_to with shape', distance.shape)
            except Exception as e:
                print('Failed reading /environment/distance_to', e)

    # If we only have wetted, derive distance
    if distance is None and wetted is not None:
        try:
            from scipy.ndimage import distance_transform_edt
            mask = (wetted != -9999) & (wetted > 0)
            # try to approximate pixel width from coords if available
            pixel_width = 1.0
            distance = distance_transform_edt(mask) * pixel_width
            print('Computed distance raster from wetted; shape', distance.shape)
        except Exception as e:
            print('Failed to compute distance from wetted:', e)

    if distance is None:
        print('No distance raster available; aborting')
        sys.exit(0)

    # compute a target affine if possible
    target_affine = None
    try:
        if x_coords is not None and y_coords is not None and getattr(x_coords,'ndim',0) == 2 and getattr(y_coords,'ndim',0) == 2:
            coords = np.column_stack((x_coords.flatten(), y_coords.flatten()))
            target_affine = compute_affine_from_hecras(coords)
            print('Derived affine from x_coords/y_coords')
        else:
            # try HECRAS cells center coordinate
            try:
                cells = hf['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
                target_affine = compute_affine_from_hecras(cells)
                print('Derived affine from Cells Center Coordinate dataset')
            except Exception as e:
                print('Could not derive affine from HECRAS Cells dataset:', e)
    except Exception as e:
        print('Affine derivation failed:', e)

    # run extract
    try:
        main_centerline, all_lines = derive_centerline_from_distance_raster(distance, transform=target_affine, footprint_size=5, min_length=10)
        print('Extraction result: main_centerline=', bool(main_centerline))
        if main_centerline is not None:
            print('Main length:', main_centerline.length)
        print('Num lines returned:', len(all_lines))
    except Exception as e:
        print('Centerline extraction failed:', e)

print('Done')
