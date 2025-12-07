"""tin_by_timestep.py
Analyze HECRAS time series: plot wetted cell count vs time and build TIN+perimeter for a chosen timestep.
Usage:
    python tools/tin_by_timestep.py --hdf <path> [--time-index IDX] [--depth 1e-5]

If --time-index is omitted, uses the middle timestep.
"""
import os, sys, argparse
import numpy as np
import h5py

try:
    from scipy.spatial import Delaunay, cKDTree
except Exception:
    Delaunay = None; cKDTree = None

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf', required=True)
    p.add_argument('--time-index', type=int, default=None)
    p.add_argument('--depth', type=float, default=1e-5)
    p.add_argument('--max-nodes', type=int, default=5000)
    args = p.parse_args()

    if not os.path.exists(args.hdf):
        print('HDF not found:', args.hdf); sys.exit(1)

    with h5py.File(args.hdf, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
        depth_ds = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
        # depth_ds expected shape (T, N)
        depths = np.array(depth_ds[:])

    T = depths.shape[0]
    N = depths.shape[1]
    print('Time steps:', T, 'cells:', N)

    # compute wetted counts per timestep
    depth_thresh = args.depth
    wetted_counts = np.sum(depths > depth_thresh, axis=1)

    out_dir = os.path.join(os.path.dirname(args.hdf), '..', '..', 'outputs')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # plot wetted counts
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(np.arange(T), wetted_counts, '-o', markersize=3)
    ax.set_xlabel('Timestep index')
    ax.set_ylabel('Wetted cell count')
    ax.set_title(f'Wetted cells vs time (depth > {depth_thresh})')
    fig.savefig(os.path.join(out_dir, 'wetted_cells_over_time.png'), dpi=150)
    plt.close(fig)
    print('Saved', os.path.join(out_dir, 'wetted_cells_over_time.png'))

    # pick time index
    if args.time_index is None:
        ti = T // 2
    else:
        ti = args.time_index
    if ti < 0 or ti >= T:
        print('time-index out of range'); sys.exit(1)
    print('Selected timestep index:', ti)

    depth = depths[ti, :]
    mask = depth > depth_thresh
    pts = coords[mask][:, :2]
    vals = depth[mask]
    print('Wetted cells at timestep', ti, ':', len(pts))

    pts_s, vals_s = sample_evenly(pts, vals, max_nodes=args.max_nodes, grid_dim=120)
    print('Sampled points:', len(pts_s))

    if Delaunay is None:
        print('scipy.spatial.Delaunay required')
        sys.exit(1)
    tri = Delaunay(pts_s)
    tris = tri.simplices
    verts = np.column_stack([pts_s[:,0], pts_s[:,1], vals_s])
    np.savez_compressed(os.path.join(out_dir, 'tin_experiment_bytime.npz'), verts=verts, faces=tris, values=vals_s)
    print('Saved mesh to', os.path.join(out_dir, 'tin_experiment_bytime.npz'))

    # perimeter (vector neighbor check)
    perim_pts = None
    if cKDTree is not None:
        try:
            all_coords = coords[:, :2]
            all_depth = depth
            # here use depths at that timestep
            all_depth_t = depths[ti, :]
            tree = cKDTree(all_coords)
            wetted_idx = np.where(all_depth_t > depth_thresh)[0]
            k = 8
            neigh_dists, neigh_idx = tree.query(all_coords[wetted_idx], k=k)
            min_dry_frac = 0.30
            is_perim = []
            for nbrs in neigh_idx:
                nbrs = np.array(nbrs)
                dry_count = np.sum(all_depth_t[nbrs] <= depth_thresh)
                frac_dry = float(dry_count) / max(1, len(nbrs))
                is_perim.append(frac_dry >= min_dry_frac)
            perim_pts = all_coords[wetted_idx][np.array(is_perim, dtype=bool)]
            print('Perimeter points:', 0 if perim_pts is None else perim_pts.shape[0])
        except Exception as e:
            print('Perimeter failed:', e)
            perim_pts = None

    # save PNG overlay
    fig, ax = plt.subplots(figsize=(8,8))
    ax.triplot(verts[:,0], verts[:,1], tris, linewidth=0.25, color='0.4')
    if perim_pts is not None and len(perim_pts)>0:
        ax.scatter(perim_pts[:,0], perim_pts[:,1], s=2, c='red')
    ax.set_aspect('equal')
    png = os.path.join(out_dir, f'tin_experiment_t{ti}_perim.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print('Saved', png)

if __name__ == '__main__':
    main()
