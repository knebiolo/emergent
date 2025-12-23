"""Compute simple schooling metrics from a sim HDF5 DB.

Usage:
    python tools/schooling_metrics.py /path/to/sim_db.h5

Outputs a CSV with per-timestep summary and a debug CSV with per-agent nearest-neighbor distances.
"""
import sys
import os
import h5py
import numpy as np
from scipy.spatial import cKDTree
import json
import time


def compute_metrics(db_path, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(db_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(db_path, 'r') as f:
        X = f['agent_data/X'][()]
        Y = f['agent_data/Y'][()]
        # try heading, else compute from velocities
        heading = f['agent_data/heading'][()] if 'agent_data/heading' in f else None
        x_vel = f['agent_data/x_vel'][()] if 'agent_data/x_vel' in f else None
        y_vel = f['agent_data/y_vel'][()] if 'agent_data/y_vel' in f else None

    n_agents, n_steps = X.shape[0], X.shape[1]
    summary = []

    timestamp = int(time.time())
    debug_rows = []

    for t in range(n_steps):
        xs = X[:, t]
        ys = Y[:, t]
        valid = np.isfinite(xs) & np.isfinite(ys)
        coords = np.column_stack((xs[valid], ys[valid]))
        if coords.shape[0] < 2:
            # not enough agents
            summary.append({
                'timestep': t,
                'n_valid': int(coords.shape[0]),
                'mean_nn_dist': None,
                'median_nn_dist': None,
                'min_nn_dist': None,
                'polarization': None,
                'cohesion': None,
                'mean_speed': None,
            })
            continue

        tree = cKDTree(coords)
        dists, idxs = tree.query(coords, k=2)
        # dists[:,0] is zero (self), use dists[:,1]
        nn = dists[:, 1]
        mean_nn = float(np.mean(nn))
        med_nn = float(np.median(nn))
        min_nn = float(np.min(nn))

        # cohesion: mean distance to centroid
        centroid = coords.mean(axis=0)
        coh = float(np.mean(np.linalg.norm(coords - centroid, axis=1)))

        # polarization / alignment
        if heading is not None:
            h = heading[valid, t]
            ux = np.cos(h)
            uy = np.sin(h)
            meanvec = np.array([ux.mean(), uy.mean()])
            pol = float(np.linalg.norm(meanvec) / 1.0)
        elif x_vel is not None and y_vel is not None:
            vx = x_vel[valid, t]
            vy = y_vel[valid, t]
            speeds = np.sqrt(vx**2 + vy**2)
            nz = speeds > 0
            if np.any(nz):
                ux = vx[nz] / speeds[nz]
                uy = vy[nz] / speeds[nz]
                meanvec = np.array([ux.mean(), uy.mean()])
                pol = float(np.linalg.norm(meanvec))
            else:
                pol = None
        else:
            pol = None

        # mean speed
        if x_vel is not None and y_vel is not None:
            speed_all = np.sqrt(x_vel[:, t]**2 + y_vel[:, t]**2)
            mean_speed = float(np.nanmean(speed_all))
        else:
            mean_speed = None

        summary.append({
            'timestep': t,
            'n_valid': int(coords.shape[0]),
            'mean_nn_dist': mean_nn,
            'median_nn_dist': med_nn,
            'min_nn_dist': min_nn,
            'polarization': pol,
            'cohesion': coh,
            'mean_speed': mean_speed,
        })

        # debug per-agent nn distances (map back to agent indices)
        # build an array of length n_agents with nan for invalid
        nn_full = np.full((n_agents,), np.nan, dtype=float)
        nn_full[np.where(valid)[0]] = nn
        for ai in range(n_agents):
            debug_rows.append((t, ai, float(nn_full[ai]) if not np.isnan(nn_full[ai]) else None))

    # write summary JSON and CSV
    summary_path = os.path.join(out_dir, f'schooling_summary_{timestamp}.json')
    with open(summary_path, 'w') as sf:
        json.dump(summary, sf, indent=2)

    debug_path = os.path.join(out_dir, f'schooling_debug_nn_{timestamp}.csv')
    with open(debug_path, 'w') as df:
        df.write('timestep,agent,nn_dist\n')
        for r in debug_rows:
            t, ai, nnv = r
            df.write(f'{t},{ai},{nnv if nnv is not None else ""}\n')

    print('Wrote summary to', summary_path)
    print('Wrote debug CSV to', debug_path)
    return summary_path, debug_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/schooling_metrics.py /path/to/sim_db.h5')
        sys.exit(2)
    db = sys.argv[1]
    outdir = None
    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
    compute_metrics(db, outdir)
