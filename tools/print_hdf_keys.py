import sys, os, h5py
if len(sys.argv) < 2:
    print('Usage: print_hdf_keys.py <hdf_path>')
    sys.exit(2)
path = sys.argv[1]
if not os.path.exists(path):
    print('Path not found:', path); sys.exit(2)

def walk(group, prefix=''):
    for k in group.keys():
        item = group[k]
        print(prefix + '/' + k, type(item))
        try:
            if isinstance(item, h5py.Group):
                walk(item, prefix + '/' + k)
        except Exception:
            pass

with h5py.File(path, 'r') as f:
    print('Top-level keys:', list(f.keys()))
    walk(f, '')
