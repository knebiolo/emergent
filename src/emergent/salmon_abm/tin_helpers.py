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
