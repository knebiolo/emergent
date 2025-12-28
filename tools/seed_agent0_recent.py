import h5py
import numpy as np
import sys

pre = sys.argv[1]
if len(sys.argv) > 2:
    ts = int(sys.argv[2])
else:
    ts = 50

with h5py.File(pre,'a') as f:
    if 'memory' not in f:
        mem = f.create_group('memory')
    else:
        mem = f['memory']
    # create a small 21x21 grid centered area with recent timestamps
    shape = (21,21)
    grid = np.full(shape, ts, dtype='f4')
    if '0' in mem:
        del mem['0']
    mem.create_dataset('0', data=grid, dtype='f4')
    # ensure mental_map_transform exists; use existing or create default
    if 'mental_map_transform' not in f.attrs:
        f.attrs['mental_map_transform'] = np.array([100.0,0.0,548025.636,0.0,-100.0,6642002.78966], dtype='f4')

print('Seeded agent 0 with timestamp', ts, 'in', pre)
