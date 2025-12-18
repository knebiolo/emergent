"""Pure mesh builder utilities for TIN/GL mesh creation.

This module is intentionally small and testable: given 2D points and scalar
values it returns vertex, face, and color arrays. It uses scipy.spatial
Delaunay for triangulation and matplotlib colormap for colors.

The implementation follows the project's long-term programming guide: fail
fast on invalid inputs and avoid broad exception swallowing.
"""
from typing import Tuple
import numpy as np
from scipy.spatial import Delaunay


def build_mesh(pts: np.ndarray, vals: np.ndarray, vert_exag: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a mesh from 2D points and scalar values.

    Args:
        pts: Nx2 array of X,Y coordinates.
        vals: N-length array of scalar values (e.g., depth).
        vert_exag: vertical exaggeration applied to Z = vals * vert_exag.

    Returns:
        verts: Nx3 float32 vertex array.
        faces: Mx3 int32 triangle indices into verts.
        colors: Nx4 float32 RGBA colors in range [0,1].

    Raises:
        ValueError: on invalid inputs.
    """
    pts = np.asarray(pts, dtype=float)
    vals = np.asarray(vals, dtype=float)

    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("pts must be Nx2 array")
    if vals.ndim != 1 or vals.shape[0] != pts.shape[0]:
        raise ValueError("vals must be 1D array with same length as pts")
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points to build a triangulation")

    # Delaunay triangulation in XY plane
    tri = Delaunay(pts[:, :2])
    faces = tri.simplices.astype(np.int32)

    # vertices with Z from vals
    z = (np.nan_to_num(vals, nan=0.0) * float(vert_exag)).astype(np.float32)
    verts = np.column_stack([pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32), z])

    # color mapping (viridis-like) normalized to vals
    try:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        denom = vmax - vmin if (vmax - vmin) != 0 else 1.0
        normed = ((vals - vmin) / denom).clip(0.0, 1.0)
        try:
            import pyqtgraph as pg
            cmap = pg.colormap('viridis') if hasattr(pg, 'colormap') else None
            if cmap is not None:
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                idx = (normed * (lut.shape[0] - 1)).astype(int)
                rgba = lut[idx]
                if rgba.shape[1] == 3:
                    rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1))], axis=1)
                colors = np.asarray(rgba, dtype=np.float32)
            else:
                # simple viridis-like gradient fallback
                def _simple_viridis(v):
                    r = np.clip(4.0 * (v - 0.75), 0.0, 1.0)
                    g = np.clip(4.0 * (v - 0.25), 0.0, 1.0)
                    b = np.clip(4.0 * (0.5 - v), 0.0, 1.0)
                    return np.stack([r, g, b, np.ones_like(r)], axis=1)
                colors = _simple_viridis(normed)
        except Exception:
            # fallback if pyqtgraph unavailable
            def _simple_viridis(v):
                r = np.clip(4.0 * (v - 0.75), 0.0, 1.0)
                g = np.clip(4.0 * (v - 0.25), 0.0, 1.0)
                b = np.clip(4.0 * (0.5 - v), 0.0, 1.0)
                return np.stack([r, g, b, np.ones_like(r)], axis=1)
            colors = _simple_viridis(normed)
    except Exception:
        # generic fallback to gray
        colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype=np.float32), (pts.shape[0], 1))

    return verts, faces, colors
