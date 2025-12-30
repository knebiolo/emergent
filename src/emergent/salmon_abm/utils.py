"""Utility helpers for salmon_abm migration.

These are intentionally small, well-documented, and dependency-light so
unit tests can exercise them without heavy GIS stacks.
"""
from typing import Any, Tuple, Union
import numpy as np


def _is_transform_like(obj: Any) -> bool:
    if obj is None:
        return False
    if hasattr(obj, "a") and hasattr(obj, "c") and hasattr(obj, "e") and hasattr(obj, "f"):
        return True
    try:
        seq = tuple(obj)
    except Exception:
        return False
    # Be strict: a transform should look like a 6-tuple (GDAL) or 3x3 Affine (9),
    # not an arbitrary long coordinate vector.
    if len(seq) not in (6, 9):
        return False
    try:
        # ensure elements are scalar numbers
        for v in seq:
            if hasattr(v, "shape") and getattr(v, "shape", ()) != ():
                return False
            float(v)
    except Exception:
        return False
    return True



def _unpack_affine(transform):
    """Return (a,b,c,d,e,f) from a transform object or tuple.

    Accepts either an object with attributes `a,b,c,d,e,f` (rasterio.Affine)
    or a 6-tuple/list.
    """
    if transform is None:
        raise ValueError("transform must be provided")
    # Prefer attribute access but be permissive: allow missing b/d (assume 0)
    if hasattr(transform, "a") and hasattr(transform, "c") and hasattr(transform, "e") and hasattr(transform, "f"):
        a = getattr(transform, "a")
        b = getattr(transform, "b", 0.0)
        c = getattr(transform, "c")
        d = getattr(transform, "d", 0.0)
        e = getattr(transform, "e")
        f = getattr(transform, "f")
        return (a, b, c, d, e, f)

    try:
        a, b, c, d, e, f = transform
    except Exception as exc:
        raise ValueError("Unsupported transform format") from exc
    return (a, b, c, d, e, f)


def geo_to_pixel(x: float, y: float, transform) -> Tuple[int, int]:
    """Convert geospatial coordinates (x, y) to pixel indices (row, col).

    Uses the affine transform convention: x = a*col + b*row + c,
    y = d*col + e*row + f.
    """
    a, b, c, d, e, f = _unpack_affine(transform)
    # Solve linear system for (col, row): [a b; d e] [col; row] = [x-c; y-f]
    # If transform supports inverse multiplication (~transform * (x,y)), prefer it
    inv = None
    if hasattr(transform, "__invert__"):
        try:
            inv = ~transform
        except Exception:
            inv = None

    if inv is not None:
        # use inverse mapping; support scalar or iterable inputs
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        cols = []
        rows = []
        for xi, yi in zip(x_arr, y_arr):
            col_i, row_i = inv * (xi, yi)
            cols.append(col_i)
            rows.append(row_i)
        cols = np.asarray(cols, dtype=float)
        rows = np.asarray(rows, dtype=float)
        # sanitize any NaN/inf results
        cols = np.nan_to_num(cols, nan=0.0, posinf=0.0, neginf=0.0)
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        # inverse mapping likely already accounts for pixel-center offsets
        if rows.size == 1:
            return int(np.rint(rows[0])), int(np.rint(cols[0]))
        return np.rint(rows).astype(int), np.rint(cols).astype(int)

    # fallback: numeric solver and explicit pixel-center handling
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    # Fast path for common north-up rasters (no shear/rotation).
    if b == 0 and d == 0 and a != 0 and e != 0:
        col = (x_arr - c) / a
        row = (y_arr - f) / e
    else:
        A = np.array([[a, b], [d, e]], dtype=float)
        rhs = np.vstack([x_arr - c, y_arr - f])
        sol = np.linalg.solve(A, rhs)
        col = sol[0]
        row = sol[1]
    # Use pixel-center convention: convert to pixel indices for pixel centers
    col = col - 0.5
    row = row - 0.5
    # sanitize NaN/inf and convert to ints safely
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    if row.size == 1:
        return int(np.floor(row[0] + 0.5)), int(np.floor(col[0] + 0.5))
    return np.floor(row + 0.5).astype(int), np.floor(col + 0.5).astype(int)


def pixel_to_geo(row: int, col: int, transform) -> Tuple[float, float]:
    """Convert pixel indices (row, col) to geospatial coordinates (x, y).

    Backwards-compatible: some legacy callers pass arguments as
    (transform, row, col). This function detects that form and accepts
    both orders.
    """
    # Detect if caller passed (transform, row, col)
    if _is_transform_like(row) and not _is_transform_like(transform):
        transform, row, col = row, col, transform

    a, b, c, d, e, f = _unpack_affine(transform)
    # vectorized handling: accept scalars or arrays for row/col
    row_arr = np.atleast_1d(np.asarray(row, dtype=float))
    col_arr = np.atleast_1d(np.asarray(col, dtype=float))
    x = a * col_arr + b * row_arr + c
    y = d * col_arr + e * row_arr + f
    # return scalars when inputs were scalars
    if x.size == 1:
        return float(x[0]), float(y[0])
    return x, y


def standardize_shape(arr_or_shape: Union[np.ndarray, Tuple[int, int]], target_shape=None, fill_value=np.nan):
    """Return shape or pad/crop array depending on arguments.

    - If `target_shape` is None and `arr_or_shape` is an array, returns its
      `(rows, cols)` tuple.
    - If `target_shape` is provided, treats `arr_or_shape` as an array and
      returns a padded/cropped array of shape `target_shape` filled with
      `fill_value` where necessary.
    - If `arr_or_shape` is a 2-tuple and `target_shape` is None, returns it.
    """
    if target_shape is None:
        if isinstance(arr_or_shape, tuple) and len(arr_or_shape) == 2:
            return arr_or_shape
        if hasattr(arr_or_shape, 'shape'):
            return tuple(arr_or_shape.shape[:2])
        raise ValueError("Input must be a 2D array or shape tuple")

    # target_shape provided: perform pad/crop
    arr = arr_or_shape
    if not hasattr(arr, 'shape'):
        raise ValueError("arr must be an array when target_shape is provided")
    tr, tc = target_shape
    r, c = arr.shape[:2]
    out = np.full((tr, tc), fill_value, dtype=arr.dtype)
    nr = min(tr, r)
    nc = min(tc, c)
    out[:nr, :nc] = arr[:nr, :nc]
    return out


def standardize_shape_pad(arr: np.ndarray, target_shape=(5, 5), fill_value=np.nan) -> np.ndarray:
    """Pad or crop `arr` to `target_shape`. Returns a new array.

    - If `arr` is smaller, it will be placed at the top-left of the target and
      padded with `fill_value`.
    - If `arr` is larger, it will be cropped to the target.
    """
    if not hasattr(arr, 'shape'):
        raise ValueError("arr must be an array")
    tr, tc = target_shape
    r, c = arr.shape[:2]
    out = np.full((tr, tc), fill_value, dtype=arr.dtype)
    nr = min(tr, r)
    nc = min(tc, c)
    out[:nr, :nc] = arr[:nr, :nc]
    return out


def determine_slices(center: Tuple[int, int], half_size: Union[int, Tuple[int, int]], shape: Tuple[int, int]):
    """Return row/col slices centered at `center` with `half_size`, clipped to `shape`.

    - `center` is (row, col)
    - `half_size` may be int or (half_rows, half_cols)
    - `shape` is (nrows, ncols)
    """
    r, c = center
    if isinstance(half_size, int):
        hr = hc = half_size
    else:
        hr, hc = half_size
    nrows, ncols = shape
    r0 = max(0, r - hr)
    r1 = min(nrows, r + hr + 1)
    c0 = max(0, c - hc)
    c1 = min(ncols, c + hc + 1)
    return slice(r0, r1), slice(c0, c1)


def calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0):
    """Return masks indicating whether grid cells are in front of agents.

    - `headings`: array-like of shape (n_agents,) in radians
    - `x_coords`, `y_coords`: arrays shaped (n_agents, H, W) giving cell coords
    - `agent_x`, `agent_y`: arrays shaped (n_agents,) agent positions

    Returns array of shape (n_agents, H, W) with 1 in front, `behind_value` otherwise.
    """
    headings = np.asarray(headings)

    # Accept x_coords/y_coords as either (H, W) or (n_agents, H, W)
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    if x_coords.ndim == 2:
        # broadcast to per-agent arrays
        x_coords = np.broadcast_to(x_coords, (headings.shape[0],) + x_coords.shape)
        y_coords = np.broadcast_to(y_coords, (headings.shape[0],) + y_coords.shape)

    dx = np.cos(headings)[:, np.newaxis, np.newaxis]
    dy = np.sin(headings)[:, np.newaxis, np.newaxis]
    agent_x_expanded = np.asarray(agent_x)[:, np.newaxis, np.newaxis]
    agent_y_expanded = np.asarray(agent_y)[:, np.newaxis, np.newaxis]
    rel_x = x_coords - agent_x_expanded
    rel_y = y_coords - agent_y_expanded
    dot_product = dx * rel_x + dy * rel_y
    front_masks = (dot_product > 0).astype(int)
    front_masks[dot_product <= 0] = behind_value
    return front_masks


def determine_slices_from_vectors(vectors, num_slices=4):
    """Map a set of vectors (N x 2) to slice indices (0..num_slices-1)."""
    vectors = np.asarray(vectors)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    normalized_angles = np.mod(angles, 2 * np.pi)
    slice_width = 2 * np.pi / num_slices
    slice_indices = (normalized_angles // slice_width).astype(int)
    return slice_indices


def determine_slices_from_headings(headings, num_slices=4):
    """Map headings (radians) to discrete slice indices (0..num_slices-1)."""
    headings = np.mod(np.asarray(headings), 2 * np.pi)
    slice_width = 2 * np.pi / num_slices
    slice_indices = (headings // slice_width).astype(int)
    return slice_indices


def linear_interpolate(a: Union[float, np.ndarray], b: Union[float, np.ndarray], frac: float):
    """Linear interpolation between `a` and `b` by fraction `frac`."""
    a = np.array(a)
    b = np.array(b)
    return a + frac * (b - a)


def calculate_front_mask(values: np.ndarray, axis: int = 0, threshold: float = None) -> np.ndarray:
    """Compute a simple front mask based on absolute gradient along `axis`.

    - If `threshold` is None, use mean + std of the abs-gradient as threshold.
    - Returns a boolean array the same shape as `values` where True indicates
        a front (high gradient) at that cell.
    """
    if values.ndim < 1:
        raise ValueError("values must be an array")
    grad = np.abs(np.diff(values, axis=axis))
    # pad to original shape
    pad_shape = list(values.shape)
    pad_shape[axis] = 1
    pad = np.zeros(tuple(pad_shape), dtype=grad.dtype)
    grad_full = np.concatenate([grad, pad], axis=axis)
    if threshold is None:
        threshold = grad_full.mean() + grad_full.std()
    return grad_full > threshold


__all__ = [
    'geo_to_pixel',
    'pixel_to_geo',
    'standardize_shape',
    'determine_slices',
    'standardize_shape_pad',
    'determine_slices_from_vectors',
    'determine_slices_from_headings',
    'calculate_front_masks',
    'linear_interpolate',
    'calculate_front_mask',
]

