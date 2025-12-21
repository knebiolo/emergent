import numpy as np
from rasterio.transform import Affine
from scipy.spatial import cKDTree


def compute_affine_from_hecras(coords, target_cell_size=None):
    """Compute a conservative Affine transform from HECRAS cell center coordinates.

    Strategy:
    - Compute nearest-neighbor distances for a random subset of points and take the median spacing.
    - Use that spacing as the `x`/`y` pixel size (square cells).
    - Use the min-x and max-y of coords as the origin (upper-left corner), adjusting by half-cell.

    Returns an `Affine` suitable for rasterizing / geo_to_pixel mapping.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    n = coords.shape[0]
    # sample up to 2000 points for spacing calc
    sample_n = min(2000, n)
    idx = np.random.choice(n, size=sample_n, replace=False)
    sample = coords[idx]

    # build KDTree and get 2nd NN distances (first is zero/self)
    tree = cKDTree(coords)
    dists, _ = tree.query(sample, k=2)
    # second column are nearest neighbor distances
    nn = dists[:, 1]
    # median spacing
    median_spacing = float(np.median(nn))
    if target_cell_size is not None:
        # prefer requested target_cell_size if provided but don't exceed median spacing
        cell = float(target_cell_size)
    else:
        cell = max(median_spacing, 1e-6)

    minx = float(coords[:, 0].min())
    maxy = float(coords[:, 1].max())
    # use minx, maxy as upper-left corner but shift by half-cell to center cells
    origin_x = minx - 0.5 * cell
    origin_y = maxy + 0.5 * cell

    return Affine(cell, 0.0, origin_x, 0.0, -cell, origin_y)


def geo_to_pixel_from_inv(inv, X, Y):
    """Convert coordinates to pixel indices using precomputed inverse affine `inv`.

    `inv` is expected to have attributes a,b,c,d,e,f (an Affine object).
    """
    xs = np.asarray(X, dtype=float)
    ys = np.asarray(Y, dtype=float)
    cols = inv.c + inv.a * (xs + 0.0) + inv.b * (ys + 0.0)
    rows = inv.f + inv.d * (xs + 0.0) + inv.e * (ys + 0.0)
    return np.rint(rows).astype(int), np.rint(cols).astype(int)


def get_inv_transform(sim, transform):
    """Return cached inverse affine for `transform` on `sim`.

    Caches by id(transform) to avoid repeated Affine inversion costs.
    """
    try:
        key = id(transform)
    except Exception:
        return ~transform
    cache = getattr(sim, '_inv_transform_cache', None)
    if cache is None:
        cache = {}
        sim._inv_transform_cache = cache
    inv = cache.get(key)
    if inv is None:
        try:
            inv = ~transform
        except Exception:
            # best-effort: return direct invertible object
            return ~transform
        cache[key] = inv
    return inv


def geo_to_pixel(X, Y, transform):
    """
    Convert x, y coordinates to row, column indices in the raster grid.
    This function inverts the provided affine transform to convert geographic
    coordinates to pixel coordinates.

    Parameters:
    - X: array-like of x coordinates (longitude or projected x)
    - Y: array-like of y coordinates (latitude or projected y)
    - transform: affine transform of the raster

    Returns:
    - rows: array of row indices
    - cols: array of column indices
    """
    # Try to use vectorized affine math if transform exposes coefficients
    try:
        inv = get_inv_transform(getattr(transform, '__self__', None) or globals().get('sim', None), transform)
        # inv is an Affine; compute cols, rows via inv.c + inv.a*(x+0.5) + inv.b*(y+0.5)
        xs = np.asarray(X, dtype=float)
        ys = np.asarray(Y, dtype=float)
        # Affine multiplication for inverse: (col, row) = inv * (x, y)
        cols = inv.c + inv.a * (xs + 0.0) + inv.b * (ys + 0.0)
        rows = inv.f + inv.d * (xs + 0.0) + inv.e * (ys + 0.0)
        rows = np.rint(rows).astype(int)
        cols = np.rint(cols).astype(int)
        return rows, cols
    except Exception:
        # Fallback to per-point multiplication
        inv_transform = ~transform
        pixels = [inv_transform * (x, y) for x, y in zip(X, Y)]
        cols, rows = zip(*pixels)
        rows = np.round(rows).astype(int)
        cols = np.round(cols).astype(int)
        return rows, cols
