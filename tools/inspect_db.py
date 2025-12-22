import os
import glob
import h5py
import numpy as np

out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
patterns = glob.glob(os.path.join(out_dir, 'sim_db_*.h5'))
if not patterns:
    print('No sim DB files found in', out_dir)
    raise SystemExit(1)

latest = max(patterns, key=os.path.getmtime)
print('Using DB:', latest)

with h5py.File(latest, 'r') as f:
    def show(name, ds):
        try:
            arr = ds[:]
            print(name, 'shape=', arr.shape, 'dtype=', arr.dtype, 'nan_count=', int(np.isnan(arr).sum()) if arr.size else 0, 'min=', np.nanmin(arr) if arr.size else None, 'max=', np.nanmax(arr) if arr.size else None)
        except Exception as e:
            print(name, '<non-array or error>', e)

    # top-level
    print('\nTop-level keys:')
    for k in f.keys():
        print(' -', k)

    if 'environment' in f:
        print('\nEnvironment datasets:')
        for k in f['environment'].keys():
            show('environment/'+k, f['environment'][k])

    if 'x_coords' in f:
        show('x_coords', f['x_coords'])
    if 'y_coords' in f:
        show('y_coords', f['y_coords'])

    if 'X' in f:
        show('X', f['X'])
    if 'Y' in f:
        show('Y', f['Y'])

print('\nDone')
