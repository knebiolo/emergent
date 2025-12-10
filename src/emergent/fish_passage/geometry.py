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
"""
geometry.py

Preamble/Module plan for geometric utilities (moved to fish_passage).

Responsibilities (planned):
- Functions for coordinate transforms, pixel<->geo conversions, affine computations.
- Utilities for perimeter simplification, point-in-polygon checks, and centerline derivation helpers.
- Small helpers should be pure functions with clear input validation and documented bounds.

Notes:
- Keep functions short (<60 lines) and well-tested.
"""
