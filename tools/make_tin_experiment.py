"""make_tin_experiment.py

Standalone script to read a HECRAS HDF plan, sample wetted points,
compute a Delaunay TIN, and save the mesh to outputs/tin_experiment.npz
and outputs/tin_experiment.png for quick inspection.

This script is intentionally self-contained and does not modify the
repo aside from writing outputs. Delete this file after you're satisfied
with the TIN and we've integrated the final code.
"""
import os
import sys
import argparse
import numpy as np

try:
    from scipy.spatial import Delaunay, cKDTree
except Exception as e:
    raise RuntimeError('scipy.spatial required')

try:
    import h5py
except Exception:
    raise RuntimeError('h5py required')

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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


def build_tin(hdf_path, depth_thresh=0.05, max_nodes=5000, grid_dim=120, out_dir='outputs'):
    with h5py.File(hdf_path, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
    mask = depth > depth_thresh
    pts = coords[mask][:, :2]
    vals = depth[mask]
    print(f'Wetted cells: {len(pts)}')
    pts, vals = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=grid_dim)
    print(f'Sampled points: {len(pts)}')
    if len(pts) < 3:
        raise RuntimeError('Not enough points to triangulate')
    tri = Delaunay(pts)
    tris = tri.simplices
    z = vals * 1.0
    verts = np.column_stack([pts[:,0], pts[:,1], z])
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, 'tin_experiment.npz'), verts=verts, faces=tris, values=vals)
    print('Saved mesh to', os.path.join(out_dir, 'tin_experiment.npz'))
    # compute wetted perimeter (vector approach using neighbor checks)
    perim_pts = None
    try:
        from scipy.spatial import cKDTree
        all_coords = coords[:, :2]
        all_depth = depth
        wetted_idx = np.where(all_depth > depth_thresh)[0]
        if len(wetted_idx) > 0:
            tree = cKDTree(all_coords)
            k = 8
            neigh_dists, neigh_idx = tree.query(all_coords[wetted_idx], k=k)
            # relaxed rule: require a fraction of neighbors to be dry
            is_perim = []
            min_dry_frac = 0.30
            for nbrs in neigh_idx:
                nbrs = np.array(nbrs)
                dry_count = np.sum(all_depth[nbrs] <= depth_thresh)
                frac_dry = float(dry_count) / max(1, len(nbrs))
                is_perim.append(frac_dry >= min_dry_frac)
            perim_mask = np.array(is_perim, dtype=bool)
            perim_pts = all_coords[wetted_idx][perim_mask]
            print('Perimeter points:', 0 if perim_pts is None else perim_pts.shape[0])
    except Exception as e:
        print('Perimeter computation failed:', e)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.triplot(verts[:,0], verts[:,1], tris, linewidth=0.25, color='0.4')
        if perim_pts is not None and len(perim_pts) > 0:
            ax.scatter(perim_pts[:,0], perim_pts[:,1], s=2, c='red', alpha=0.8, label='perimeter')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        fig.tight_layout()
        png_path = os.path.join(out_dir, 'tin_experiment_perim.png')
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print('Saved preview PNG with perimeter to', png_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--hdf', required=False, help='Path to HECRAS .p05.hdf file')
    p.add_argument('--depth', type=float, default=0.05)
    p.add_argument('--max-nodes', type=int, default=5000)
    p.add_argument('--grid-dim', type=int, default=120)
    args = p.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.hdf:
        hdf_path = args.hdf
    else:
        data_dir = os.path.join(repo_root, 'data', 'salmon_abm', '20240506')
        hdf_path = None
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.p05.hdf'):
                    hdf_path = os.path.join(data_dir, f)
                    break
    if hdf_path is None or not os.path.exists(hdf_path):
        print('HECRAS .p05.hdf not found; pass --hdf')
        sys.exit(1)
    build_tin(hdf_path, depth_thresh=args.depth, max_nodes=args.max_nodes, grid_dim=args.grid_dim, out_dir=os.path.join(repo_root, 'outputs'))
