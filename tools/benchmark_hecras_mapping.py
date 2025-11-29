import time
import h5py
import numpy as np
from scipy.spatial import cKDTree
from numpy.linalg import lstsq

plan = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"

print('Loading HECRAS cell centers and sample field...')
with h5py.File(plan, 'r') as h:
    coords = h['/Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:].astype(np.float32)
    # pick a scalar field: Cells Minimum Elevation
    field = h['/Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'][:]

# Filter to cells with finite field values (avoid dry cells / nodata)
finite_mask = np.isfinite(field)
coords_valid = coords[finite_mask]
field_valid = field[finite_mask].astype(np.float64)

n = coords.shape[0]
print('n cells =', n)
print('valid cells =', coords_valid.shape[0])

# build KDTree
print('Building KDTree...')
start = time.time()
tree = cKDTree(coords_valid)
print('KDTree built in %.3fs' % (time.time() - start))

# sample test indices (avoid extreme memory): pick 2000 random cells
rng = np.random.default_rng(12345)
valid_n = coords_valid.shape[0]
test_idx = rng.choice(valid_n, size=min(2000, valid_n), replace=False)
query_pts = coords_valid[test_idx]
true_vals = field_valid[test_idx]

# Method A: inverse-distance weighted k-NN
def map_idw(query, k=3, eps=1e-8):
    dists, inds = tree.query(query, k=k)
    # if k==1 shapes differ
    if k == 1:
        dists = dists[:, None]
        inds = inds[:, None]
    inv = 1.0 / (dists + eps)
    w = inv / np.sum(inv, axis=1)[:, None]
    vals = field_valid[inds]
    mapped = np.sum(vals * w, axis=1)
    return mapped

# Method B: local linear (moving least-squares) fit using k neighbors
# For each query, fit linear model: value ~ a + b*(x-x0) + c*(y-y0) using neighbors and weights
def map_local_linear(query, k=16, eps=1e-8, reg=1e-8):
    dists, inds = tree.query(query, k=k)
    if k == 1:
        dists = dists[:, None]
        inds = inds[:, None]
    mapped = np.empty((query.shape[0],), dtype=np.float64)
    for i in range(query.shape[0]):
        qi = query[i]
        neigh = coords_valid[inds[i]]
        vals = field_valid[inds[i]]
        # weights: inverse distance
        w = 1.0 / (dists[i] + eps)
        # build design matrix for local linearization around qi
        A = np.column_stack([np.ones(k), neigh[:, 0] - qi[0], neigh[:, 1] - qi[1]]).astype(np.float64)
        # weighted least squares: solve (W A)x = W y where W = diag(sqrt(w))
        sqrtw = np.sqrt(w)
        Aw = A * sqrtw[:, None]
        yw = vals * sqrtw
        # regularize by adding reg to the diagonal of (A^T A)
        try:
            coef, *_ = lstsq(Aw, yw, rcond=None)
            mapped[i] = coef[0]
        except Exception:
            inv = 1.0 / (dists[i] + eps)
            ww = inv / np.sum(inv)
            mapped[i] = np.sum(vals * ww)
    return mapped

# Run benchmarks
for k_idw in (3, 8):
    t0 = time.time()
    mapped = map_idw(query_pts, k=k_idw)
    dt = time.time() - t0
    # RMSE on finite outputs
    valid = np.isfinite(mapped) & np.isfinite(true_vals)
    if np.any(valid):
        rmse = np.sqrt(np.mean((mapped[valid] - true_vals[valid])**2))
    else:
        rmse = np.nan
    print('IDW k=%d: time=%.3fs, rmse=%s' % (k_idw, dt, np.format_float_positional(rmse, precision=4)))

for k_lin in (8, 16):
    t0 = time.time()
    mapped = map_local_linear(query_pts, k=k_lin)
    dt = time.time() - t0
    valid = np.isfinite(mapped) & np.isfinite(true_vals)
    if np.any(valid):
        rmse = np.sqrt(np.mean((mapped[valid] - true_vals[valid])**2))
    else:
        rmse = np.nan
    print('LocalLinear k=%d: time=%.3fs, rmse=%s' % (k_lin, dt, np.format_float_positional(rmse, precision=4)))

print('Done')
