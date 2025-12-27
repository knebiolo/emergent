#!/usr/bin/env python3
"""Inspect HDF5 diagnostics and produce a small summary JSON and printed report.

Usage: python tools/inspect_h5_cues.py <h5_path> [--step 0] [--outdir outputs/diagnostics/cue_checks]
"""
import argparse
import h5py
import numpy as np
import json
from pathlib import Path

def vec_angle_deg(v):
    return np.degrees(np.arctan2(v[:,1], v[:,0]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('h5')
    p.add_argument('--step', type=int, default=0)
    p.add_argument('--outdir', default='outputs/diagnostics/cue_checks')
    args = p.parse_args()
    h5p = args.h5
    step = str(args.step)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5p, 'r') as f:
        if 'steps' not in f or step not in f['steps']:
            print('No steps/', step, 'in', h5p)
            return 2
        g = f['steps'][step]
        keys = list(g.keys())
        print('Found keys in steps/%s:' % step, keys)
        # identify per-agent vec keys
        cue_keys = [k for k in keys if k.endswith('_vec')]
        # filter to (n,2)
        good_cues = []
        for k in cue_keys:
            arr = np.array(g[k])
            if arr.ndim == 2 and arr.shape[1] == 2:
                good_cues.append(k)
        print('Per-agent cue vec keys:', good_cues)

        payload = {k: np.array(g[k]) for k in good_cues}
        if 'head_vec' in g:
            head = np.array(g['head_vec'])
            if head.ndim == 1 and head.size % 2 == 0:
                head = head.reshape(-1,2)
        else:
            head = None

        n_agents = None
        if payload:
            n_agents = next(iter(payload.values())).shape[0]
        print('n_agents inferred:', n_agents)

        # compute per-cue mags and mean
        cue_stats = {}
        for k, arr in payload.items():
            mags = np.linalg.norm(arr, axis=1)
            cue_stats[k] = {'mean_mag': float(np.nanmean(mags)), 'median_mag': float(np.nanmedian(mags)), 'max_mag': float(np.nanmax(mags))}

        # compute resultant and compare to head
        if payload:
            mats = [payload[k].astype(float) for k in payload.keys()]
            resultant = np.sum(np.stack(mats, axis=0), axis=0)
            res_angles = vec_angle_deg(resultant)
        else:
            resultant = None
            res_angles = None

        if head is not None and resultant is not None and head.shape == resultant.shape:
            head_angles = vec_angle_deg(head)
            diffs = np.abs((res_angles - head_angles + 180) % 360 - 180)
            stats = {'mean_abs_diff_deg': float(np.nanmean(diffs)), 'median_abs_diff_deg': float(np.nanmedian(diffs)), 'max_abs_diff_deg': float(np.nanmax(diffs))}
        else:
            stats = {}

        # compute per-agent per-cue fractions
        contribs = []
        if resultant is not None:
            for i in range(resultant.shape[0]):
                total = float(np.linalg.norm(resultant[i]))
                for k, arr in payload.items():
                    mag = float(np.linalg.norm(arr[i]))
                    frac = float(mag / total) if total > 0 else None
                    contribs.append({'agent': int(i), 'cue': k, 'mag': mag, 'frac': frac})

    # write a small JSON report
    out = {
        'h5': h5p,
        'step': int(step),
        'n_agents': n_agents,
        'cue_stats': cue_stats,
        'resultant_stats': stats,
        'n_cue_keys': len(good_cues),
        'per_agent_contrib_count': len(contribs)
    }
    base = Path(h5p).stem
    out_json = outdir / f'{base}_step{step}_inspect.json'
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote', out_json)
    print(json.dumps(out, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
