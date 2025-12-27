import numpy as np
import os
import glob

p = 'outputs/diagnostics/forced_rawvecs'
files = sorted(glob.glob(os.path.join(p, '*.npz')))
if not files:
    print('No NPZ files found in', p)
    raise SystemExit(1)
for f in files:
    print('File:', f)
    try:
        d = np.load(f)
        print('  keys=', d.files)
        for k in d.files:
            v = d[k]
            try:
                print('   -', k, 'shape=', getattr(v, 'shape', None), 'dtype=', getattr(v, 'dtype', None))
            except Exception as e:
                print('   -', k, 'read-error', e)
    except Exception as e:
        print('  Failed to read', f, 'err=', e)
    print()
