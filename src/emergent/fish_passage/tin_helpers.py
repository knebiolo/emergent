import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import polygonize, unary_union


def sample_evenly(points, n_samples):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= n_samples:
        return pts.copy()
    idx = np.linspace(0, pts.shape[0] - 1, n_samples, dtype=int)
    return pts[idx]


def alpha_shape(points, alpha=1.0):
    """Compute a concave hull (alpha-shape) from a set of 2D points.

    This is a lightweight implementation using Delaunay triangulation and
    filtering edges by circumradius threshold derived from `alpha`.
    Returns a shapely Polygon (unioned) or None if insufficient points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 4:
        # fallback: convex hull via Polygon
        try:
            poly = Polygon(pts).convex_hull
            return poly
        except Exception:
            return None

    tri = Delaunay(pts)
    triangles = pts[tri.simplices]
    # compute circumradius for each triangle
    def circumradius(tri_pts):
        a = np.linalg.norm(tri_pts[1] - tri_pts[0])
        b = np.linalg.norm(tri_pts[2] - tri_pts[1])
        c = np.linalg.norm(tri_pts[0] - tri_pts[2])
        s = 0.5 * (a + b + c)
        area = max(1e-12, np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c))))
        return (a * b * c) / (4.0 * area)

    radii = np.array([circumradius(t) for t in triangles])
    keep = radii < (1.0 / max(1e-12, float(alpha)))
    edges = []
    for tri_idx, t in enumerate(tri.simplices):
        if not keep[tri_idx]:
            continue
        i, j, k = t
        edges.append((tuple(pts[i]), tuple(pts[j])))
        edges.append((tuple(pts[j]), tuple(pts[k])))
        edges.append((tuple(pts[k]), tuple(pts[i])))

    if not edges:
        return None

    mls = MultiLineString(edges)
    merged = unary_union(mls)
    polys = list(polygonize(merged))
    if not polys:
        return None
    poly = unary_union(polys)
    return poly
