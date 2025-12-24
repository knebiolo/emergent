"""Inspect movement debug NPZ files and summarize arrays.

Usage:
  python tools/inspect_move_debug.py

This script searches `outputs/diagnostics` for `move_debug_step_*.npz`,
loads the newest file, computes simple diagnostics and writes a JSON
summary `outputs/diagnostics/move_debug_summary.json`.
"""
import os
import glob
import json
import numpy as np


def find_latest_npz(out_dir='outputs/diagnostics'):
    pattern = os.path.join(out_dir, 'move_debug_step_*.npz')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def summarize_npz(fn):
    try:
        data = np.load(fn)
    except Exception as e:
        return {'error': f'failed to load {fn}: {e}'}

    summary = {'file': fn, 'arrays': {}}
    for k in data.files:
        arr = data[k]
        try:
            shape = arr.shape
            size = int(arr.size if hasattr(arr, 'size') else 0)
            unique = None
            if arr.size > 0 and arr.dtype.kind in ('f','i','u','b'):
                try:
                    # flatten floats to limited precision to avoid tiny differences
                    if np.issubdtype(arr.dtype, np.floating):
                        unique = int(np.unique(np.round(arr, 8)).size)
                    else:
                        unique = int(np.unique(arr).size)
                except Exception:
                    unique = None
            stats = {}
            if arr.dtype.kind in ('f','i','u','b') and arr.size > 0:
                flat = arr.astype(float).ravel()
                stats['min'] = float(np.nanmin(flat))
                stats['max'] = float(np.nanmax(flat))
                stats['mean'] = float(np.nanmean(flat))
                stats['std'] = float(np.nanstd(flat))
            summary['arrays'][k] = {'shape': shape, 'size': size, 'unique_values_estimate': unique, 'stats': stats}
        except Exception as e:
            summary['arrays'][k] = {'error': str(e)}
    return summary


def main():
    out_dir = os.path.join('outputs', 'diagnostics')
    latest = find_latest_npz(out_dir)
    if latest is None:
        print('No movement debug NPZ files found in', out_dir)
        return 1
    summary = summarize_npz(latest)
    out_fn = os.path.join(out_dir, 'move_debug_summary.json')
    with open(out_fn, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote summary to', out_fn)
    # print key highlights
    arrays = summary.get('arrays', {})
    if 'dxdy' in arrays:
        dxdy_info = arrays['dxdy']
        print('dxdy shape:', dxdy_info.get('shape'), 'unique_est:', dxdy_info.get('unique_values_estimate'))
    if 'thrust' in arrays:
        print('thrust shape:', arrays['thrust'].get('shape'))
    if 'drag' in arrays:
        print('drag shape:', arrays['drag'].get('shape'))
    return 0


if __name__ == '__main__':
    rc = main()
    raise SystemExit(rc)
