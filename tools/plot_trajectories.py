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
    # show depth raster using geographic extents so trajectories overlay correctly
    # x_coords and y_coords are raster-shaped arrays giving world coordinates for each pixel
    x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
    y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))
    extent = [x_min, x_max, y_min, y_max]
    # use a blue colormap reversed so deeper = darker
    im = ax.imshow(depth, cmap='Blues_r', origin='upper', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label='Depth (m)')

    # Plot trajectories directly in world coordinates (X, Y). Split on NaNs and large jumps.
    n_agents = X.shape[0]
    n_steps = X.shape[1] if X.ndim > 1 else 1
    max_jump = max(x_max - x_min, y_max - y_min) * 0.1  # large jump threshold (10% of domain)
    segments = []
    starts = []
    for ai in range(n_agents):
        xs = X[ai] if X.ndim > 1 else np.array([X[ai]])
        ys = Y[ai] if Y.ndim > 1 else np.array([Y[ai]])
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(valid) < 2:
            continue
        idxs_valid = np.where(valid)[0]
        # split on non-consecutive timesteps or large jumps in world units
        run_start = idxs_valid[0]
        last_idx = run_start
        run_x = [xs[run_start]]
        run_y = [ys[run_start]]
        for k in idxs_valid[1:]:
            jump = np.hypot(xs[k] - xs[last_idx], ys[k] - ys[last_idx])
            if k != last_idx + 1 or jump > max_jump:
                # end current run
                if len(run_x) >= 2:
                    segments.append(np.column_stack((run_x, run_y)))
                    starts.append((run_x[0], run_y[0]))
                run_x = [xs[k]]
                run_y = [ys[k]]
            else:
                run_x.append(xs[k])
                run_y.append(ys[k])
            last_idx = k
        if len(run_x) >= 2:
            segments.append(np.column_stack((run_x, run_y)))
            starts.append((run_x[0], run_y[0]))
    # add all segments as a LineCollection
    if segments:
        # convert geographic segments to pixel coords for plotting over the image; but since image uses extent,
        # plotting world coords directly is fine
        lc = LineCollection(segments, linewidths=0.6, colors='black', alpha=0.6)
        ax.add_collection(lc)
        starts = np.array(starts)
        if starts.size:
            ax.scatter(starts[:, 0], starts[:, 1], s=4, c='red')

    ax.set_title('Agent trajectories over depth raster')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_aspect('equal', adjustable='box')
    # no legend for readability when many agents
    plt.tight_layout()
    fig.savefig(out_path)
    print('Saved plot to', out_path)


if __name__ == '__main__':
    db = find_latest_db()
    out_fn = os.path.join(OUT, 'trajectories.png')
    plot_db(db, out_fn)
