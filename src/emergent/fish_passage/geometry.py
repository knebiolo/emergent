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


def project_points_onto_line(xs_line, ys_line, px, py):
    """Project points (px,py) onto a polyline defined by xs_line, ys_line.

    Returns distances along the polyline for each point (float array).
    This is a numpy-vectorized implementation suitable for tests and small
    to medium-sized inputs.
    """
    xs_line = np.asarray(xs_line, dtype=float)
    ys_line = np.asarray(ys_line, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

    S = xs_line.size - 1
    if S <= 0 or px.size == 0:
        return np.zeros(px.size, dtype=float)

    seg_x0 = xs_line[:-1]
    seg_y0 = ys_line[:-1]
    seg_x1 = xs_line[1:]
    seg_y1 = ys_line[1:]
    vx = seg_x1 - seg_x0
    vy = seg_y1 - seg_y0
    seg_len = np.hypot(vx, vy)
    seg_len_safe = np.where(seg_len == 0, 1e-12, seg_len)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Broadcast shapes: M x S
    M = px.size
    px_e = px[:, None]
    py_e = py[:, None]
    x0_e = seg_x0[None, :]
    y0_e = seg_y0[None, :]
    vx_e = vx[None, :]
    vy_e = vy[None, :]

    wx = px_e - x0_e
    wy = py_e - y0_e
    denom = vx_e * vx_e + vy_e * vy_e
    denom = np.where(denom == 0, 1e-12, denom)
    t = (wx * vx_e + wy * vy_e) / denom
    t_clamped = np.clip(t, 0.0, 1.0)
    cx = x0_e + t_clamped * vx_e
    cy = y0_e + t_clamped * vy_e
    d2 = (px_e - cx) ** 2 + (py_e - cy) ** 2
    idx = np.argmin(d2, axis=1)
    chosen_t = t_clamped[np.arange(M), idx]
    chosen_seg = idx
    distances_along = cumlen[chosen_seg] + chosen_t * seg_len[chosen_seg]
    return distances_along


def compute_distance_to_bank(coords, wetted_mask, perimeter_indices, median_spacing=None):
    """Compute distance-to-bank on irregular mesh via graph Dijkstra.

    Parameters
    - coords: (N,2) array of point coordinates
    - wetted_mask: boolean array of length N indicating wetted cells
    - perimeter_indices: list/array of indices into coords marking perimeter cells
    - median_spacing: optional precomputed spacing; if None, estimate from random sample

    Returns:
    - distances_all: (N,) float array where perimeter indices are 0 and non-wetted are NaN
    """
    coords = np.asarray(coords, dtype=float)
    wetted_mask = np.asarray(wetted_mask, dtype=bool)
    n = coords.shape[0]
    # estimate median spacing if needed
    if median_spacing is None:
        sample_size = min(1000, len(coords))
        if sample_size <= 0:
            median_spacing = 1.0
        else:
            sample_idx = np.random.choice(len(coords), size=sample_size, replace=False)
            sample_coords = coords[sample_idx]
            try:
                from emergent.fish_passage.utils import safe_build_kdtree
                sample_tree = safe_build_kdtree(sample_coords, name='hecras_sample_tree')
            except Exception:
                sample_tree = None
            if sample_tree is not None:
                dists, _ = sample_tree.query(sample_coords, k=2)
                median_spacing = float(np.median(dists[:, 1]))
            else:
                bbox = sample_coords.max(axis=0) - sample_coords.min(axis=0)
                median_spacing = float(np.sqrt((bbox[0] * bbox[1]) / max(1, sample_size)))

    wetted_coords = coords[wetted_mask]
    wetted_indices = np.where(wetted_mask)[0]
    n_wetted = len(wetted_coords)
    try:
        from emergent.fish_passage.utils import safe_build_kdtree
        wetted_tree = safe_build_kdtree(wetted_coords, name='hecras_wetted_tree')
    except Exception:
        wetted_tree = None
    if wetted_tree is None:
        return np.full(len(coords), np.nan, dtype=np.float32)

    connectivity_radius = median_spacing * 1.5
    pairs = wetted_tree.query_pairs(r=connectivity_radius, output_type='ndarray')
    if len(pairs) > 0:
        row = pairs[:, 0]
        col = pairs[:, 1]
        edge_coords_i = wetted_coords[row]
        edge_coords_j = wetted_coords[col]
        edge_dists = np.sqrt(np.sum((edge_coords_i - edge_coords_j) ** 2, axis=1))
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        data_sym = np.concatenate([edge_dists, edge_dists])
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        graph = csr_matrix((data_sym, (row_sym, col_sym)), shape=(n_wetted, n_wetted))
        perimeter_wetted_indices = []
        for perim_idx in perimeter_indices:
            pos = np.where(wetted_indices == perim_idx)[0]
            if len(pos) > 0:
                perimeter_wetted_indices.append(pos[0])
        perimeter_wetted_indices = np.array(perimeter_wetted_indices, dtype=np.int32)
        if len(perimeter_wetted_indices) > 0:
            dist_matrix = dijkstra(csgraph=graph, directed=False, indices=perimeter_wetted_indices)
            if dist_matrix.ndim == 2:
                distances_wetted = np.min(dist_matrix, axis=0)
            else:
                distances_wetted = dist_matrix
        else:
            distances_wetted = np.full(n_wetted, np.inf)
    else:
        distances_wetted = np.full(n_wetted, np.inf)

    distances_all = np.full(len(coords), np.nan, dtype=np.float32)
    distances_all[wetted_indices] = distances_wetted.astype(np.float32)
    distances_all[perimeter_indices] = 0.0
    return distances_all


def order_points_nearest_neighbor(points, start_idx=None):
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if start_idx is None:
        start_idx = 0
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    order = [int(start_idx)]
    remaining = set(range(n))
    remaining.remove(int(start_idx))
    while remaining:
        cur = order[-1]
        dists, idxs = tree.query(pts[cur], k=min(len(remaining), n))
        next_idx = None
        for candidate in np.atleast_1d(idxs):
            if int(candidate) in remaining:
                next_idx = int(candidate)
                break
        if next_idx is None:
            next_idx = remaining.pop()
            order.append(next_idx)
        else:
            order.append(next_idx)
            remaining.remove(next_idx)
    return np.array(order, dtype=int)


def thin_perimeter_uniform(points, target_spacing):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return pts.copy()
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    out = [pts[0]]
    acc = 0.0
    for i in range(1, pts.shape[0]):
        d = np.linalg.norm(pts[i] - pts[i-1])
        acc += d
        if acc >= target_spacing:
            out.append(pts[i])
            acc = 0.0
    if len(out) > 1 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return np.array(out, dtype=float)

