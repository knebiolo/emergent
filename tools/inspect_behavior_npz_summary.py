#!/usr/bin/env python3
"""Inspect full behavior debug NPZ files (not alignment sidecars) and print per-cue summaries.

Usage: python tools/inspect_behavior_npz_summary.py --dir outputs/diagnostics
"""
import argparse
import glob
import numpy as np
import math
import os


def mean_angle_deg(vecs):
    if vecs is None or getattr(vecs, 'size', 0) == 0:
        return float('nan')
    vx = np.nanmean(np.asarray(vecs)[:, 0])
    vy = np.nanmean(np.asarray(vecs)[:, 1])
    ang = math.degrees(math.atan2(vy, vx))
    return ang


def summarize(fname):
    d = np.load(fname)
    print('\nFile:', os.path.basename(fname))
    cues = [k for k in d.files if k.endswith('_vec')]
    for cue in sorted(cues):
        arr = np.asarray(d[cue]).astype(float)
        if arr.size == 0:
            mag = float('nan')
            ang = float('nan')
        else:
            if arr.ndim == 1:
                mag = float(np.nanmean(np.abs(arr)))
                ang = float('nan')
            else:
                mag = float(np.nanmean(np.linalg.norm(arr, axis=1)))
                ang = mean_angle_deg(arr)
        print(f'  {cue:30s} mean_mag={mag:8.4g} mean_ang_deg={ang:7.2f}')
    if 'head_vec' in d.files:
        hv = np.asarray(d['head_vec']).astype(float)
        if hv.size:
            print('  head_vec mean_mag=', float(np.nanmean(np.linalg.norm(hv, axis=1))), ' mean_ang_deg=', mean_angle_deg(hv))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', default='outputs/diagnostics')
    args = p.parse_args()
    files = sorted([f for f in glob.glob(os.path.join(args.dir, 'behavior_debug_step_*.npz')) if not f.endswith('_alignment.npz')])
    if not files:
        print('No behavior_debug_step_*.npz files found in', args.dir)
        return 2
    # pick a few sample steps (first, middle, last)
    sample = [files[0]]
    if len(files) > 2:
        sample.append(files[len(files)//2])
    if len(files) > 1:
        sample.append(files[-1])
    for f in sample:
        try:
            summarize(f)
        except Exception as e:
            print('Failed to summarize', f, e)


if __name__ == '__main__':
    raise SystemExit(main())
