"""Plot agent trajectories over depth raster from the latest sim DB.

Usage:
    python tools/plot_trajectories.py

Saves: outputs/trajectories.png
"""
import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
PAT = os.path.join(OUT, 'sim_db_*.h5')


def find_latest_db():
    files = glob.glob(PAT)
    if not files:
        raise FileNotFoundError('No sim_db_*.h5 found in outputs/')
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_db(db_path, out_path):
    with h5py.File(db_path, 'r') as f:
        if 'environment/depth' not in f:
            raise KeyError('depth raster not found in DB')
        depth = f['environment/depth'][()]
        x_coords = f['environment/x_coords'][()]
        y_coords = f['environment/y_coords'][()]
        # agent positions are stored in agent_data/X and agent_data/Y (shape: n_agents, n_steps?)
        if 'agent_data/X' in f and 'agent_data/Y' in f:
            X = f['agent_data/X'][()]
            Y = f['agent_data/Y'][()]
        else:
            # fall back to agent_data prev/cur datasets
            X = f['agent_data/X'][()]
            Y = f['agent_data/Y'][()]

    fig, ax = plt.subplots(figsize=(10, 8))
    # show depth raster
    im = ax.imshow(depth, cmap='viridis', origin='upper')
    fig.colorbar(im, ax=ax, label='Depth')

    n_agents = X.shape[0]
    n_steps = X.shape[1] if X.ndim > 1 else 1
    # Vectorized mapping: build KDTree of pixel coordinates for fast nearest-neighbor lookup
    # x_coords/y_coords are raster-shaped arrays (height, width)
    h, w = depth.shape
    pts = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    tree = cKDTree(pts)

    # Query KDTree once for all agent-time coordinates (vectorized)
    n_steps = X.shape[1] if X.ndim > 1 else 1
    coords_flat = np.column_stack((X.reshape(-1), Y.reshape(-1)))
    valid_flat = np.isfinite(coords_flat[:, 0]) & np.isfinite(coords_flat[:, 1])
    idxs_flat = np.full(coords_flat.shape[0], -1, dtype=int)
    if np.any(valid_flat):
        _, idxs_valid = tree.query(coords_flat[valid_flat], k=1)
        idxs_flat[valid_flat] = idxs_valid
    idxs = idxs_flat.reshape(n_agents, n_steps)
    rows = (idxs // w).astype(int)
    cols = (idxs % w).astype(int)

    # Build segments per-agent but split segments on gaps or large pixel jumps to avoid long connectors
    max_gap = 50  # pixels; tune if needed
    segments = []
    starts = []
    for ai in range(n_agents):
        idx_ai = idxs[ai]
        valid_mask = idx_ai != -1
        if np.count_nonzero(valid_mask) < 2:
            continue
        valid_positions = np.where(valid_mask)[0]
        # compute pixel coordinates for valid positions
        c = cols[ai, valid_positions]
        r = rows[ai, valid_positions]
        # compute pixel distances between successive valid positions
        dc = np.diff(c)
        dr = np.diff(r)
        pdist = np.sqrt(dc.astype(float)**2 + dr.astype(float)**2)
        # a connection is allowed when the valid positions are consecutive timesteps and pixel jump small
        consecutive = np.diff(valid_positions) == 1
        connect = consecutive & (pdist <= max_gap)
        # split into runs where connect is True
        run_starts = [0]
        for i, ok in enumerate(connect):
            if not ok:
                run_starts.append(i + 1)
        # now build segments from run_starts
        for start in run_starts:
            # find end index: start .. next break where connect False
            j = start
            while j < len(valid_positions) - 1 and connect[j]:
                j += 1
            seg_c = c[start:j + 1]
            seg_r = r[start:j + 1]
            if seg_c.size >= 2:
                seg = np.column_stack((seg_c, seg_r))
                segments.append(seg)
                starts.append((seg_c[0], seg_r[0]))
    if segments:
        lc = LineCollection(segments, linewidths=1.0, colors='black', alpha=0.9)
        ax.add_collection(lc)
        starts = np.array(starts)
        if starts.size:
            ax.scatter(starts[:, 0], starts[:, 1], s=4, c='red')

    ax.set_title('Agent trajectories over depth raster')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    # no legend for readability when many agents
    plt.tight_layout()
    fig.savefig(out_path)
    print('Saved plot to', out_path)


if __name__ == '__main__':
    db = find_latest_db()
    out_fn = os.path.join(OUT, 'trajectories.png')
    plot_db(db, out_fn)
