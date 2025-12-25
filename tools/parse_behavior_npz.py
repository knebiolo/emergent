#!/usr/bin/env python3
"""Parse behavior debug NPZ files created by `behavior.arbitrate()` and
produce a compact CSV summarizing per-cue mean magnitude and mean angle.

Usage: python tools/parse_behavior_npz.py --dir outputs/diagnostics --out summary.csv
"""
import argparse
import os
import numpy as np
import math
import glob


def mean_angle(vecs):
    # vecs: (n,2)
    if vecs is None or getattr(vecs, 'size', 0) == 0:
        return float('nan')
    vx = np.asarray(vecs)[:, 0]
    vy = np.asarray(vecs)[:, 1]
    ang = math.atan2(np.nanmean(vy), np.nanmean(vx))
    return ang


def summarize_npz(fname):
    data = np.load(fname)
    out = {'file': os.path.basename(fname)}
    # cues are saved as '<cue>_vec' and '<cue>_mag'
    for key in data.files:
        if key.endswith('_vec'):
            cue = key[:-4]
            arr = np.asarray(data[key]).astype(float)
            if arr.ndim == 1:
                # per-agent scalar replicated
                magn = np.nanmean(arr)
                ang = float('nan')
            else:
                magn = float(np.nanmean(np.linalg.norm(arr, axis=1)))
                ang = mean_angle(arr)
            out[f'{cue}_mean_mag'] = magn
            out[f'{cue}_mean_ang'] = ang
    # also include head_vec
    if 'head_vec' in data.files:
        hv = np.asarray(data['head_vec']).astype(float)
        out['head_vec_mean_mag'] = float(np.nanmean(np.linalg.norm(hv, axis=1))) if hv.size else float('nan')
        out['head_vec_mean_ang'] = mean_angle(hv) if hv.size else float('nan')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', default='outputs/diagnostics')
    p.add_argument('--out', '-o', default='outputs/diagnostics/behavior_summary.csv')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, 'behavior_debug_step_*.npz')))
    if not files:
        print('No behavior_debug_step_*.npz files found in', args.dir)
        return 2

    rows = []
    for f in files:
        try:
            rows.append(summarize_npz(f))
        except Exception as e:
            print('Failed parsing', f, e)

    # collect columns
    cols = set()
    for r in rows:
        cols.update(r.keys())
    cols = sorted(cols)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(c, '')) for c in cols) + '\n')

    print('Wrote summary to', args.out)


if __name__ == '__main__':
    raise SystemExit(main())
