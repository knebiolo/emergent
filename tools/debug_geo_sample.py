import os
import glob
import h5py
import numpy as np
from emergent.salmon_abm.utils import geo_to_pixel

out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
patterns = glob.glob(os.path.join(out_dir, 'sim_db_*.h5'))
if not patterns:
    print('No sim DB files found in', out_dir)
    raise SystemExit(1)
latest = max(patterns, key=os.path.getmtime)
print('Using DB:', latest)
with h5py.File(latest, 'r') as f:
    depth = f['environment/depth'][:]
    x_coords_env = f['environment/x_coords'][:]
    y_coords_env = f['environment/y_coords'][:]
    X = f['X'][:]
    Y = f['Y'][:]

    # estimate affine (assumes mostly north-up but supports rotation)
    # solve for a,b,c in x = a*col + b*row + c using three points
    nrows, ncols = depth.shape
    # pick reference points
    x00 = x_coords_env[0,0]
    x10 = x_coords_env[1,0]
    x01 = x_coords_env[0,1]
    y00 = y_coords_env[0,0]
    y10 = y_coords_env[1,0]
    y01 = y_coords_env[0,1]
    # equations:
    # x00 = a*0 + b*0 + c -> c = x00
    # x01 = a*1 + b*0 + c -> a = x01 - c
    # x10 = a*0 + b*1 + c -> b = x10 - c
    c = x00
    a = x01 - c
    b = x10 - c
    f_ = y00
    d = y01 - f_
    e = y10 - f_
    affine = (a, b, c, d, e, f_)
    print('Estimated affine:', affine)

    # compute pixel indices
    rows, cols = geo_to_pixel(X, Y, affine)
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)
    print('rows min/max:', rows.min(), rows.max(), 'cols min/max:', cols.min(), cols.max())
    valid = (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
    print('valid mask (first 20):', valid[:20])
    print('valid count:', valid.sum(), 'out of', X.size)
    # show sample values
    out = np.full(X.size, np.nan)
    if np.any(valid):
        out[valid] = depth[rows[valid], cols[valid]]
    print('sample depth first 10:', out[:10])
    # also show top-left area
    print('depth[0:3,0:3]=\n', depth[:3,:3])

print('done')
