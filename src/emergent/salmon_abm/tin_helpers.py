import numpy as np
from scipy.spatial import cKDTree

def sample_evenly(pts, vals, max_nodes=5000, grid_dim=100):
    """Sample points evenly across the spatial extent using a grid-based selection.

    - pts: Nx2 array of coordinates
    - vals: N array of associated values
    - max_nodes: maximum number of points to return
    - grid_dim: number of grid cells per axis to use for stratified sampling

    Returns: sampled_pts, sampled_vals
    """
    pts = np.asarray(pts)
    vals = np.asarray(vals)
    if len(pts) <= max_nodes:
        return pts, vals

    # create grid over extent
    minx, miny = np.min(pts, axis=0)
    maxx, maxy = np.max(pts, axis=0)
    xs = np.linspace(minx, maxx, grid_dim+1)
    ys = np.linspace(miny, maxy, grid_dim+1)

    buckets = [[] for _ in range(grid_dim*grid_dim)]

    ix = np.searchsorted(xs, pts[:,0], side='right') - 1
    iy = np.searchsorted(ys, pts[:,1], side='right') - 1
    ix = np.clip(ix, 0, grid_dim-1)
    iy = np.clip(iy, 0, grid_dim-1)

    for i, (xi, yi) in enumerate(zip(ix, iy)):
        buckets[xi*grid_dim + yi].append(i)

    selected = []
    per_bucket = max(1, int(np.ceil(max_nodes / (grid_dim*grid_dim))))
    # first pass: sample up to per_bucket from each non-empty bucket
    for b in buckets:
        if len(b) == 0:
            continue
        if len(b) <= per_bucket:
            selected.extend(b)
        else:
            rng = np.random.default_rng(0)
            selected.extend(rng.choice(b, size=per_bucket, replace=False).tolist())

    # if too many selected, downsample uniformly
    if len(selected) > max_nodes:
        rng = np.random.default_rng(1)
        selected = rng.choice(selected, size=max_nodes, replace=False).tolist()

    sampled_pts = pts[selected]
    sampled_vals = vals[selected]
    return sampled_pts, sampled_vals


def alpha_shape(points, alpha=None):
    """Compute the alpha shape (concave hull) of a set of points.

    Returns a Shapely polygon (may be MultiPolygon) or None on failure.
    If alpha is None, a heuristic based on median edge length is used.
    """
    try:
        import numpy as _np
        from scipy.spatial import Delaunay
        from shapely.geometry import MultiLineString, Polygon, MultiPoint
        from shapely.ops import polygonize, unary_union
    except Exception:
        return None


def triangulate_and_clip(pts, vals, poly=None, max_nodes=5000, grid_dim=120):
    """Triangulate points and clip triangles with an optional Shapely polygon.

    - pts: (N,2) point coordinates
    - vals: (N,) scalar values associated (e.g., depth)
    - poly: Shapely geometry used to clip triangles (optional)
    - max_nodes: maximum number of points to use (will sample evenly)
    - grid_dim: grid dimension for sampling

    Returns: verts (M,3), faces (K,3)
    """
    pts = np.asarray(pts)
    vals = np.asarray(vals)
    if pts.shape[0] == 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=int)

    if pts.shape[0] > max_nodes:
        pts_s, vals_s = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=grid_dim)
    else:
        pts_s, vals_s = pts, vals

    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pts_s[:, :2])
        faces = tri.simplices
        verts = np.column_stack([pts_s[:, 0], pts_s[:, 1], vals_s])
    except Exception:
        return np.column_stack([pts_s[:, 0], pts_s[:, 1], vals_s]), np.zeros((0, 3), dtype=int)

    if poly is None or len(faces) == 0:
        return verts, faces

    # compute face centroids for clipping
    face_centroids = np.mean(verts[faces], axis=1)[:, :2]

    # Try vectorized contains (Shapely 2.x)
    try:
        from shapely import vectorized
        mask = vectorized.contains(poly, face_centroids[:, 0], face_centroids[:, 1])
        faces_clipped = faces[mask]
        return verts, faces_clipped
    except Exception:
        pass

    # Try prepared geometry with bbox prefilter
    try:
        from shapely.prepared import prep
        p = prep(poly)
        xs = face_centroids[:, 0]
        ys = face_centroids[:, 1]
        minx, miny, maxx, maxy = poly.bounds
        in_bbox = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
        idxs = np.nonzero(in_bbox)[0]
        keep = np.zeros(len(faces), dtype=bool)
        from shapely.geometry import Point
        for i in idxs:
            try:
                keep[i] = bool(p.contains(Point(float(xs[i]), float(ys[i]))))
            except Exception:
                keep[i] = False
        faces_clipped = faces[keep]
        return verts, faces_clipped
    except Exception:
        pass

    # Last-resort per-point check
    try:
        from shapely.geometry import Point
        keep = []
        for c in face_centroids:
            try:
                keep.append(bool(poly.contains(Point(float(c[0]), float(c[1])))))
            except Exception:
                keep.append(False)
        keep = np.array(keep, dtype=bool)
        faces_clipped = faces[keep]
        return verts, faces_clipped
    except Exception:
        return verts, np.zeros((0, 3), dtype=int)

    pts = _np.asarray(points)
    if pts.shape[0] < 4:
        # too few points for concave hull; return convex hull of points
        try:
            return MultiPoint(list(map(tuple, pts))).convex_hull
        except Exception:
            return None

    try:
        tri = Delaunay(pts)
    except Exception:
        return None

    triangles = pts[tri.simplices]

    # compute circumradius for each triangle
    a = _np.linalg.norm(triangles[:,0] - triangles[:,1], axis=1)
    b = _np.linalg.norm(triangles[:,1] - triangles[:,2], axis=1)
    c = _np.linalg.norm(triangles[:,2] - triangles[:,0], axis=1)
    s = (a + b + c) / 2.0
    # area via Heron
    area = _np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    with _np.errstate(invalid='ignore'):
        circum_r = (a * b * c) / (4.0 * _np.sqrt(area))

    # heuristic alpha: median circumradius * factor
    if alpha is None:
        finite_r = circum_r[_np.isfinite(circum_r)]
        if finite_r.size == 0:
            alpha = 1.0
        else:
            median_r = _np.median(finite_r)
            alpha = median_r * 1.5

    # keep triangles with circumradius <= alpha
    mask = (circum_r <= alpha)
    edges = []
    for tri_idx, keep in enumerate(mask):
        if not keep:
            continue
        tri_pts = tri.simplices[tri_idx]
        edges.append((tuple(pts[tri_pts[0]]), tuple(pts[tri_pts[1]])))
        edges.append((tuple(pts[tri_pts[1]]), tuple(pts[tri_pts[2]])))
        edges.append((tuple(pts[tri_pts[2]]), tuple(pts[tri_pts[0]])))

    if not edges:
        return None

    try:
        m = MultiLineString(edges)
        triangles = list(polygonize(m))
        union = unary_union(triangles)
        return union
    except Exception:
        return None
