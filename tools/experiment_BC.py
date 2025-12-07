"""experiment_BC.py
Combined B+C: radius-limited neighbor fraction + small-cluster removal using KDTree BFS.
Saves outputs/tin_experiment_BC.npz and outputs/tin_experiment_BC_perim.png
"""
import os, sys
import numpy as np
try:
    from scipy.spatial import Delaunay, cKDTree
except Exception:
    raise
import h5py
import matplotlib.pyplot as plt


def sample_evenly(pts, vals, max_nodes=5000, grid_dim=120):
    pts = np.asarray(pts)
    vals = np.asarray(vals)
    if len(pts) <= max_nodes:
        return pts, vals
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
    rng = np.random.default_rng(0)
    for b in buckets:
        if len(b) == 0:
            continue
        if len(b) <= per_bucket:
            selected.extend(b)
        else:
            selected.extend(rng.choice(b, size=per_bucket, replace=False).tolist())
    if len(selected) > max_nodes:
        selected = rng.choice(selected, size=max_nodes, replace=False).tolist()
    return pts[selected], vals[selected]


def find_clusters(points, eps):
    # simple KDTree-based BFS clustering: connect points within eps
    tree = cKDTree(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = tree.query_ball_point(points[u], r=eps)
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        clusters.append(np.array(comp, dtype=int))
    return clusters


def run(hdf_path, out_dir):
    depth_thresh = 0.05
    with h5py.File(hdf_path, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
    mask = depth > depth_thresh
    pts_all = coords[mask][:, :2]
    vals_all = depth[mask]
    print('Wetted cells:', len(pts_all))
    pts, vals = sample_evenly(pts_all, vals_all, max_nodes=5000, grid_dim=120)
    print('Sampled points:', len(pts))
    tri = Delaunay(pts)
    tris = tri.simplices
    verts = np.column_stack([pts[:,0], pts[:,1], vals])
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'tin_experiment_BC.npz'), verts=verts, faces=tris, values=vals)
    # radius-limited neighbor fraction
    try:
        all_xy = coords[:, :2]
        tree = cKDTree(all_xy)
        # median NN distance
        dists, idxs = tree.query(all_xy, k=2)
        d_med = np.median(dists[:,1])
        r = 1.5 * d_med
        wetted_idx = np.where(depth > depth_thresh)[0]
        is_perim = []
        min_dry_frac = 0.30
        for wi in wetted_idx:
            nbrs = tree.query_ball_point(all_xy[wi], r=r)
            if len(nbrs) == 0:
                is_perim.append(False)
                continue
            dry_count = np.sum(depth[nbrs] <= depth_thresh)
            frac_dry = float(dry_count) / max(1, len(nbrs))
            is_perim.append(frac_dry >= min_dry_frac)
        perim_pts = all_xy[wetted_idx][np.array(is_perim, dtype=bool)]
        print('Perim raw count:', 0 if perim_pts is None else perim_pts.shape[0])
        # cluster and remove small clusters
        if perim_pts is not None and len(perim_pts) > 0:
            clusters = find_clusters(perim_pts, eps=r*1.1)
            keep_idx = []
            min_cluster_size = 50
            for comp in clusters:
                if len(comp) >= min_cluster_size:
                    keep_idx.extend(comp.tolist())
            if len(keep_idx) == 0:
                perim_pts = np.empty((0,2))
            else:
                perim_pts = perim_pts[np.array(keep_idx, dtype=int)]
            print('Perim after cluster filter:', len(perim_pts))
    except Exception as e:
        print('Perimeter failed:', e)
        perim_pts = None
    # plot
    fig, ax = plt.subplots(figsize=(10,8))
    ax.triplot(verts[:,0], verts[:,1], tris, linewidth=0.25, color='0.4')
    if perim_pts is not None and len(perim_pts)>0:
        ax.scatter(perim_pts[:,0], perim_pts[:,1], s=2, c='red')
    ax.set_aspect('equal')
    png = os.path.join(out_dir, 'tin_experiment_BC_perim.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('Saved', png)

if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo_root, 'data', 'salmon_abm', '20240506')
    hdf_path = None
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith('.p05.hdf'):
                hdf_path = os.path.join(data_dir, f)
                break
    if hdf_path is None:
        print('HECRAS HDF not found')
        sys.exit(1)
    run(hdf_path, os.path.join(repo_root, 'outputs'))