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
    # prepare georeferenced extent and mask invalid values (sentinel -9999)
    depth_mask = depth.astype(float)
    depth_mask[depth_mask <= -1000] = np.nan
    x_min, x_max = float(np.nanmin(x_coords)), float(np.nanmax(x_coords))
    y_min, y_max = float(np.nanmin(y_coords)), float(np.nanmax(y_coords))
    extent = [x_min, x_max, y_min, y_max]
    # decide origin based on y_coords ordering: if y decreases with row index, origin='upper'
    origin = 'lower'
    try:
        if np.nanmean(y_coords[0, :]) > np.nanmean(y_coords[-1, :]):
            origin = 'upper'
        else:
            origin = 'lower'
    except Exception:
        origin = 'lower'
    try:
        vmin, vmax = np.nanpercentile(depth_mask, [1, 99])
    except Exception:
        vmin, vmax = None, None
    im = ax.imshow(depth_mask, cmap='Blues_r', origin=origin, extent=extent, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label='Depth (m)')

    # Plot trajectories directly in world coordinates (X, Y). Split on NaNs and large jumps.
    n_agents = X.shape[0]
    n_steps = X.shape[1] if X.ndim > 1 else 1
    # dynamic jump threshold per agent: fraction of domain diagonal or median*4
    domain_diag = np.hypot(x_max - x_min, y_max - y_min)
    max_jump_domain = domain_diag * 0.05
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
        # compute median jump to set per-agent threshold
        if len(idxs_valid) > 2:
            jumps = np.hypot(np.diff(xs[idxs_valid]), np.diff(ys[idxs_valid]))
            med = float(np.nanmedian(jumps)) if jumps.size else 0.0
            thr = min(max_jump_domain, max(med * 4.0, 1.0))
        else:
            thr = max_jump_domain
        for k in idxs_valid[1:]:
            jump = np.hypot(xs[k] - xs[last_idx], ys[k] - ys[last_idx])
            if k != last_idx + 1 or jump > thr:
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
        # add line collection below markers (thin red lines only to reduce clutter)
            lc = LineCollection(segments, linewidths=0.4, colors='red', alpha=0.85, zorder=2)
            ax.add_collection(lc)
            starts = np.array(starts)
            if starts.size:
                # plot starts on top
                ax.scatter(starts[:, 0], starts[:, 1], s=12, c='red', zorder=3)

    ax.set_title('Agent trajectories over depth raster')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    ax.set_aspect('equal', adjustable='box')
    # no legend for readability when many agents
    plt.tight_layout()
    # save at higher DPI for clarity
    fig.savefig(out_path, dpi=300)
    print('Saved plot to', out_path)


if __name__ == '__main__':
    db = find_latest_db()
    out_fn = os.path.join(OUT, 'trajectories.png')
    plot_db(db, out_fn)
