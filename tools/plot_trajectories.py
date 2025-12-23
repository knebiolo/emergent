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
    # map geo coords to pixel index by nearest match in x_coords/y_coords
    # make flattened coordinate arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    def geo_to_pixel_single(xg, yg):
        # find nearest pixel via argmin of squared distance
        d2 = (x_flat - xg) ** 2 + (y_flat - yg) ** 2
        idx = int(np.argmin(d2))
        r = idx // depth.shape[1]
        c = idx % depth.shape[1]
        return r, c

    for ai in range(n_agents):
        xs = X[ai, :]
        ys = Y[ai, :]
        pixs = [geo_to_pixel_single(xg, yg) for xg, yg in zip(xs, ys)]
        rows = [p[0] for p in pixs]
        cols = [p[1] for p in pixs]
        ax.plot(cols, rows, marker='o', linewidth=1, markersize=2, label=f'agent{ai}')

    ax.set_title('Agent trajectories over depth raster')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    fig.savefig(out_path)
    print('Saved plot to', out_path)


if __name__ == '__main__':
    db = find_latest_db()
    out_fn = os.path.join(OUT, 'trajectories.png')
    plot_db(db, out_fn)
