"""Inspect HECRAS HDF environment group and rasters for debugging centerline extraction."""
import os, sys, h5py, numpy as np
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

hdf_path = os.path.join(repo_root, 'hecras_run.h5')
if not os.path.exists(hdf_path):
    print('HDF not found:', hdf_path)
    sys.exit(2)

with h5py.File(hdf_path, 'r') as hf:
    print('Top-level keys:', list(hf.keys()))
    env = hf.get('environment')
    if env is None:
        print('No environment group present in HDF')
        sys.exit(0)
    print('\nEnvironment keys:')
    for k in env.keys():
        try:
            ds = env[k]
            shape = getattr(ds, 'shape', None)
            dtype = getattr(ds, 'dtype', None)
            print(f' - {k}: shape={shape} dtype={dtype}')
            if shape is not None and hasattr(ds, '__iter__'):
                try:
                    arr = np.array(ds)
                    print('   stats: min=', np.nanmin(arr), 'max=', np.nanmax(arr), 'mean=', np.nanmean(arr), 'finite_count=', np.isfinite(arr).sum())
                except Exception as e:
                    print('   (could not read stats)', e)
        except Exception as e:
            print(' -', k, 'error reading', e)
    # check for cells center coords
    try:
        pts = hf['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
        print('\nCells Center Coordinate shape:', pts.shape)
        print('sample pts[0:5]:', pts[:5])
    except Exception as e:
        print('\nNo Cells Center Coordinate dataset or failed to read:', e)

print('\nDone')
