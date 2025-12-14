"""Projection helpers ported from legacy sockeye._project_points_onto_line_numba.

Provides `project_points_onto_line` which returns distances-along-line for
points projected onto a polyline defined by `xs_line`/`ys_line`.

The implementation follows the numpy fallback from the legacy code to
ensure parity when numba is not available.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def project_points_onto_line(xs_line: Sequence[float], ys_line: Sequence[float], px: Sequence[float], py: Sequence[float]) -> np.ndarray:
    """Project points `(px, py)` onto the polyline `(xs_line, ys_line)`.

    Returns a 1-D array of distances along the polyline (float64) measured
    from the polyline start to the closest projected point on the polyline.

    The algorithm matches the numpy fallback behaviour of the legacy
    `_project_points_onto_line_numba` implementation.
    """
    xs_line = np.asarray(xs_line, dtype=np.float64)
    ys_line = np.asarray(ys_line, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    # segment endpoints
    seg_x0 = xs_line[:-1]
    seg_y0 = ys_line[:-1]
    seg_x1 = xs_line[1:]
    seg_y1 = ys_line[1:]
    vx = seg_x1 - seg_x0
    vy = seg_y1 - seg_y0
    seg_len = np.hypot(vx, vy)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])

    M = px.size
    # broadcast to shape (M, S)
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


if _HAS_NUMBA:
    # provide a numba-warmed variant to match legacy naming when available
    @njit(parallel=True, cache=True)
    def project_points_onto_line_numba(xs_line, ys_line, px, py):
        S = xs_line.size - 1
        seg_x0 = xs_line[:S]
        seg_y0 = ys_line[:S]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.empty(S, dtype=np.float64)
        for j in range(S):
            seg_len[j] = math.hypot(vx[j], vy[j])
        cumlen = np.empty(S + 1, dtype=np.float64)
        cumlen[0] = 0.0
        for j in range(S):
            cumlen[j + 1] = cumlen[j] + seg_len[j]

        M = px.size
        out = np.empty(M, dtype=np.float64)
        for i in prange(M):
            best_d2 = 1e308
            best_dist = 0.0
            xi = px[i]
            yi = py[i]
            for j in range(S):
                x0 = seg_x0[j]
                y0 = seg_y0[j]
                dx = vx[j]
                dy = vy[j]
                denom = dx * dx + dy * dy
                if denom == 0.0:
                    t = 0.0
                else:
                    t = ((xi - x0) * dx + (yi - y0) * dy) / denom
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                cx = x0 + t * dx
                cy = y0 + t * dy
                d2 = (xi - cx) * (xi - cx) + (yi - cy) * (yi - cy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_dist = cumlen[j] + t * seg_len[j]
            out[i] = best_dist
        return out
