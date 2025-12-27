import sys
import numpy as np
from glob import glob
import os

paths = sys.argv[1:] or ['outputs/diagnostics', 'outputs/diagnostics/forced_rawvecs']
found = []
for base in paths:
    for p in glob(os.path.join(base, '**', '*.npz'), recursive=True):
        try:
            d = np.load(p, allow_pickle=True)
            keys = list(d.keys())
            if any(k=='head_vec' or k.endswith('_vec') or '_vec' in k for k in keys):
                found.append((p, keys))
        except Exception as e:
            print('error loading', p, e)

for p, keys in found:
    print('FILE:', p)
    print('  keys:', keys)
