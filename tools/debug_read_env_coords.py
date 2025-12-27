import h5py
import numpy as np
import sys
p = sys.argv[1]
with h5py.File(p,'r') as f:
    print('Keys:', list(f.keys()))
    if 'environment/x_coords' in f:
        xc = np.array(f['environment/x_coords'])
        yc = np.array(f['environment/y_coords'])
        print('x_coords shape', xc.shape, 'y_coords shape', yc.shape)
        print('x[0,0], y[0,0] =', float(xc[0,0]), float(yc[0,0]))
        print('x center, y center =', float(xc[xc.shape[0]//2, xc.shape[1]//2]), float(yc[yc.shape[0]//2, yc.shape[1]//2]))
    else:
        print('No x_coords/y_coords datasets')
    print('Attrs:', dict(f.attrs))
