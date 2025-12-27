#!/usr/bin/env python3
"""Summarize per-step cue dominance and movement stats for a run.

Usage: python tools/summarize_run_cues.py --trace outputs/diagnostics/<run>_trace.csv --out outputs/diagnostics/<run>_cue_summary.csv

Produces CSV with per-step: step, n_agents, mean_nn_distance, heading_circ_var, and per-cue mean_mag and dominant_count.
"""
import argparse
import glob
import os
import json
import numpy as np
from scipy.spatial import cKDTree


def circular_variance_from_vel(vx, vy):
    # vx,vy arrays
    thetas_cos = np.cos(np.arctan2(vy, vx))
    thetas_sin = np.sin(np.arctan2(vy, vx))
    mean_cos = np.nanmean(thetas_cos)
    mean_sin = np.nanmean(thetas_sin)
    R = np.sqrt(mean_cos**2 + mean_sin**2)
    circ_var = 1.0 - R
    return float(circ_var)


def find_latest_npz_for_step(npz_dir, step):
    # first look for HDF5 diagnostics in the dir
    import h5py
    h5_matches = sorted(glob.glob(os.path.join(npz_dir, '*_diagnostics.h5')))
    if h5_matches:
        # return h5 path and indicate h5 found by returning a tuple
        return h5_matches[-1]
    pattern = os.path.join(npz_dir, f'behavior_debug_step_{step}_*.npz')
    files = [f for f in glob.glob(pattern) if not f.endswith('_alignment.npz')]
    if not files:
        return None
    # pick latest modified
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def summarize(trace_csv, npz_dir, out_csv):
    import csv
    # read trace CSV into memory grouped by step
    data = {}
    steps = []
    with open(trace_csv, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            t = int(float(r['timestep']))
            a = int(r['agent'])
            x = float(r['x']) if r['x'] != '' else np.nan
            y = float(r['y']) if r['y'] != '' else np.nan
            xv = float(r['x_vel']) if r['x_vel'] != '' else 0.0
            yv = float(r['y_vel']) if r['y_vel'] != '' else 0.0
            if t not in data:
                data[t] = {'agent': [], 'x': [], 'y': [], 'xv': [], 'yv': []}
                steps.append(t)
            data[t]['agent'].append(a)
            data[t]['x'].append(x)
            data[t]['y'].append(y)
            data[t]['xv'].append(xv)
            data[t]['yv'].append(yv)
    steps = sorted(steps)
    # scan a sample NPZ to get cue list
    sample_npzs = glob.glob(os.path.join(npz_dir, 'behavior_debug_step_*.npz'))
    cue_names = []
    if sample_npzs:
        try:
            import numpy as _np
            d = _np.load(sample_npzs[0])
            cue_names = [k[:-4] for k in d.files if k.endswith('_vec')]
        except Exception:
            cue_names = []
    # output header
    header = ['step', 'n_agents', 'mean_nn_distance', 'heading_circ_var']
    for cn in cue_names:
        header.append(f'{cn}_mean_mag')
        header.append(f'{cn}_dominant_count')
    # write rows
    rows = []
    for t in steps:
        agents = np.array(data[t]['agent'])
        xs = np.array(data[t]['x'])
        ys = np.array(data[t]['y'])
        xvs = np.array(data[t]['xv'])
        yvs = np.array(data[t]['yv'])
        n = agents.size
        # mean nearest-neighbor distance
        mean_nn = float('nan')
        if n > 1:
            pts = np.column_stack((xs, ys))
            try:
                tree = cKDTree(pts)
                dists, idx = tree.query(pts, k=2)
                nn = dists[:, 1]
                mean_nn = float(np.nanmean(nn))
            except Exception:
                mean_nn = float('nan')
        # circular variance of headings
        try:
            heading_var = circular_variance_from_vel(xvs, yvs)
        except Exception:
            heading_var = float('nan')
        # find NPZ for this step
        npzfile = find_latest_npz_for_step(npz_dir, t)
        cue_mean_mags = {cn: float('nan') for cn in cue_names}
        cue_dom_counts = {cn: 0 for cn in cue_names}
        if npzfile:
            try:
                import numpy as _np
                # HDF5 path returned
                if str(npzfile).endswith('.h5') or str(npzfile).endswith('.hdf5'):
                    import h5py
                    with h5py.File(npzfile, 'r') as h5:
                        grp = h5.get('steps')
                        if grp and str(t) in grp:
                            g = grp[str(t)]
                            mag_matrix = []
                            for cn in cue_names:
                                key = cn + '_vec'
                                if key in g:
                                    arr = np.asarray(g[key]).astype(float)
                                    if arr.ndim == 1:
                                        mags = np.abs(arr)
                                        if mags.size != n:
                                            mags = np.resize(mags, n)
                                    else:
                                        mags = np.linalg.norm(arr, axis=1)
                                    cue_mean_mags[cn] = float(np.nanmean(mags))
                                    mag_matrix.append(mags)
                                else:
                                    mag_matrix.append(np.full(n, np.nan))
                            if mag_matrix:
                                M = np.column_stack(mag_matrix)
                                dom_idx = np.nanargmax(np.nan_to_num(M, nan=-np.inf), axis=1)
                                for i, cn in enumerate(cue_names):
                                    cue_dom_counts[cn] = int(np.sum(dom_idx == i))
                else:
                    d = _np.load(npzfile)
                    mag_matrix = []
                    for cn in cue_names:
                        key = cn + '_vec'
                        if key in d:
                            arr = np.asarray(d[key]).astype(float)
                            if arr.ndim == 1:
                                mags = np.abs(arr)
                                if mags.size != n:
                                    mags = np.resize(mags, n)
                            else:
                                mags = np.linalg.norm(arr, axis=1)
                            cue_mean_mags[cn] = float(np.nanmean(mags))
                            mag_matrix.append(mags)
                        else:
                            mag_matrix.append(np.full(n, np.nan))
                    if mag_matrix:
                        M = np.column_stack(mag_matrix)
                        dom_idx = np.nanargmax(np.nan_to_num(M, nan=-np.inf), axis=1)
                        for i, cn in enumerate(cue_names):
                            cue_dom_counts[cn] = int(np.sum(dom_idx == i))
            except Exception:
                # couldn't parse NPZ/HDF5
                pass
        row = [t, n, mean_nn, heading_var]
        for cn in cue_names:
            row.append(cue_mean_mags.get(cn, float('nan')))
            row.append(cue_dom_counts.get(cn, 0))
        rows.append(row)
    # write CSV
    import csv
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    print('Wrote summary to', out_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True)
    p.add_argument('--npz-dir', default='outputs/diagnostics')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    trace = args.trace
    out = args.out or trace.replace('_trace.csv', '_cue_summary.csv')
    summarize(trace, args.npz_dir, out)
