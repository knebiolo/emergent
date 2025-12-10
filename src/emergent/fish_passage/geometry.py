"""
geometry.py

Small geometry helpers: conversions between pixel (row,col) indices and
geographic coordinates using affine transforms. These functions are
intended to be simple, well-tested, and free of heavy defensive nesting.

Public functions:
- `pixel_to_geo(transform, rows, cols)` -> (xs, ys)
- `geo_to_pixel(transform, X, Y)` -> (rows, cols)

"""
from typing import Tuple, Iterable
import numpy as np


def pixel_to_geo(transform, rows, cols) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raster pixel indices to geographic coordinates.

    Parameters:
    - transform: affine.Affine-like object with attributes a,b,c,d,e,f and
      supports multiplication `(transform * (col, row))`.
    - rows, cols: scalars or array-like of same shape representing row,col

    Returns: (xs, ys) numpy arrays of the same shape as input.
    """
    rows_a = np.asarray(rows)
    cols_a = np.asarray(cols)
    # Accept scalar inputs
    scalar = False
    if rows_a.shape == () and cols_a.shape == ():
        scalar = True
        rows_a = rows_a[None]
        cols_a = cols_a[None]

    # Affine expects (x=col, y=row) as input for index->geocoord multiplication
    xs = np.empty_like(rows_a, dtype=float)
    ys = np.empty_like(rows_a, dtype=float)
    # Vectorize via flat iteration for broad Affine compatibility
    it = np.nditer(rows_a, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        i = it.multi_index
        col = float(cols_a[i])
        row = float(rows_a[i])
        xw, yw = transform * (col, row)
        xs[i] = float(xw)
        ys[i] = float(yw)
        it.iternext()

    if scalar:
        return xs[0], ys[0]
    return xs, ys


def geo_to_pixel(transform, X, Y) -> Tuple[np.ndarray, np.ndarray]:
    """Convert geographic coordinates to raster pixel indices (rows, cols).

    Returns integer row, col arrays rounded to nearest integer.
    """
    # Compute inverse affine and then apply
    inv = ~transform
    X_a = np.asarray(X)
    Y_a = np.asarray(Y)
    scalar = False
    if X_a.shape == () and Y_a.shape == ():
        scalar = True
        X_a = X_a[None]
        Y_a = Y_a[None]

    rows = np.empty_like(X_a, dtype=int)
    cols = np.empty_like(X_a, dtype=int)
    it = np.nditer(X_a, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        i = it.multi_index
        x = float(X_a[i])
        y = float(Y_a[i])
        c, r = inv * (x, y)
        rows[i] = int(round(r))
        cols[i] = int(round(c))
        it.iternext()

    if scalar:
        return rows[0], cols[0]
    return rows, cols


def compute_affine_from_hecras(coords, target_cell_size=None):
    """Compute a conservative Affine transform from irregular HECRAS cell centers.

    Strategy:
    - Use a KDTree to estimate typical nearest-neighbor spacing (median of 2nd NN).
    - Use that spacing as square cell size (unless `target_cell_size` is provided).
    - Set origin at (minx - 0.5*cell, maxy + 0.5*cell) so pixel centers align.

    Returns: `affine.Affine` instance.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        from affine import Affine as _Affine
        return _Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    n = coords.shape[0]
    sample_n = min(2000, n)
    idx = np.random.choice(n, size=sample_n, replace=False)
    sample = coords[idx]

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        dists, _ = tree.query(sample, k=2)
        nn = dists[:, 1]
        median_spacing = float(np.median(nn))
    except Exception:
        # fallback: approximate from bbox area
        bbox = coords.max(axis=0) - coords.min(axis=0)
        approx_cell = float(np.sqrt((bbox[0] * bbox[1]) / max(1, n)))
        median_spacing = approx_cell

    if target_cell_size is not None:
        cell = float(target_cell_size)
    else:
        cell = max(median_spacing, 1e-6)

    minx = float(coords[:, 0].min())
    maxy = float(coords[:, 1].max())
    origin_x = minx - 0.5 * cell
    origin_y = maxy + 0.5 * cell

    from affine import Affine as _Affine
    return _Affine(cell, 0.0, origin_x, 0.0, -cell, origin_y)
def pca_primary_axis(coords):
    """Compute the primary PCA axis for a set of 2D coordinates.

    Returns a unit vector (vx, vy) pointing along the first principal component
    and the projected scalar coordinates along that axis for each input point.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return (1.0, 0.0), np.asarray([])
    # center
    mean = coords.mean(axis=0)
    X = coords - mean
    # covariance
    C = np.dot(X.T, X) / max(1, X.shape[0] - 1)
    # eigen decomposition
    vals, vecs = np.linalg.eigh(C)
    # largest eigenvalue -> index
    idx = np.argmax(vals)
    vec = vecs[:, idx]
    # ensure unit vector
    norm = np.hypot(vec[0], vec[1])
    if norm == 0:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = vec[0] / norm, vec[1] / norm
    proj = np.dot(X, np.array([vx, vy]))
    return (float(vx), float(vy)), proj


def order_by_projection(coords):
    """Return indices that order points along the primary PCA axis (ascending).

    Ties are broken by projecting onto the secondary axis.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return np.array([], dtype=int)
    (vx, vy), proj = pca_primary_axis(coords)
    # secondary axis
    sx, sy = -vy, vx
    sec = np.dot(coords - coords.mean(axis=0), np.array([sx, sy]))
    # lexsort expects keys with the primary key last; provide secondary then primary
    keys = np.vstack((sec, proj))
    order = np.lexsort(keys)
    return order
