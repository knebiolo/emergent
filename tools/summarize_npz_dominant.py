#!/usr/bin/env python3
"""Compute per-agent dominant cue from a behavior debug NPZ and write CSV.

Usage: python tools/summarize_npz_dominant.py --file <npzfile> [--out <csv>]
"""
import argparse
import numpy as np
import math
import os
import csv

def process(npzfile, outpath=None):
    d = np.load(npzfile)
    cues = sorted([k for k in d.files if k.endswith('_vec')])
    # pick only 2D vector cues
    vecs = {c: np.asarray(d[c]).astype(float) for c in cues if np.asarray(d[c]).ndim == 2}
    if not vecs:
        print('No 2D vector cues found in', npzfile)
        return 2
    N = next(iter(vecs.values())).shape[0]
    rows = [['agent', 'dominant_cue', 'dominant_mag', 'dominant_ang_deg']]
    for i in range(N):
        best = None
        bestmag = -1.0
        for c, arr in vecs.items():
            v = arr[i]
            mag = math.hypot(float(v[0]), float(v[1]))
            if mag > bestmag:
                bestmag = mag
                best = c
        ang = float('nan')
        if bestmag > 0:
            v = vecs[best][i]
            ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
        rows.append([i, best, bestmag, ang])
    if outpath is None:
        base = os.path.basename(npzfile)
        outpath = os.path.join(os.path.dirname(npzfile), base.replace('.npz', '_dominant.csv'))
    with open(outpath, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerows(rows)
    print('Wrote', outpath)
    # print quick counts
    counts = {}
    for r in rows[1:]:
        counts[r[1]] = counts.get(r[1], 0) + 1
    print('Dominant cue counts:')
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(' ', k, v)
    return 0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', '-f', required=True)
    p.add_argument('--out', '-o')
    args = p.parse_args()
    raise SystemExit(process(args.file, args.out))
