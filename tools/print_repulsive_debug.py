import h5py
import numpy as np
import sys

fpath = 'outputs/diagnostics/avoid_single_agent_debug_h5_diagnostics.h5'

try:
    with h5py.File(fpath, 'r') as f:
        print('Top-level groups:', list(f.keys()))

        found = {'v': False}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print('Dataset:', name, 'shape=', obj.shape, 'dtype=', obj.dtype)
                if 'repulsive' in name or 'repulse' in name or 'repulsive_forces' in name:
                    print('---- Sample values for', name, '----')
                    try:
                        data = obj[()]
                        print(data)
                    except Exception as e:
                        print('Error reading dataset:', e)
                    found['v'] = True
        f.visititems(visitor)
        if not found['v']:
            print('No repulsive datasets found (searched for "repulsive" in names).')
except Exception as e:
    print('Failed to open', fpath, '->', e)
    sys.exit(2)
