"""Archived perimeter helpers.

These implementations were moved from `geometry.py` on 2025-12-09 because they
were not used elsewhere in the codebase. Kept here for historical reference only.

If you decide to restore them, copy the required functions back into
`geometry.py` and update imports/tests accordingly.
"""
from typing import Any
import numpy as np


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
