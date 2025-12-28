import h5py
import numpy as np
import sys

p = sys.argv[1]
with h5py.File(p,'r') as f:
    # step 0 is stored under steps/0 or steps/000 depending on writer
    step_keys = [k for k in f.keys() if k.startswith('steps') or k.startswith('steps/')]
    # better: check for group 'steps' then dataset '0'
    if 'steps' in f and '0' in f['steps']:
        grp = f['steps']['0']
    else:
        # try older layout: top-level 'steps/0'
        try:
            grp = f['steps/0']
        except Exception:
            # fallback try dataset names at top-level
            print('Could not find steps/0 in',p)
            sys.exit(2)
    # keys inside grp should include 'avoid_mag'
    if 'avoid_mag' in grp:
        arr = np.array(grp['avoid_mag'])
        print('avoid_mag mean', float(np.mean(arr)), 'max', float(np.max(arr)), 'nonzero_count', int((arr!=0).sum()))
    else:
        print('avoid_mag not found in step 0')
