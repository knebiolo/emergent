"""Summarize a headless trace CSV and write a JSON summary.

Produces per-timestep mean/std for x/y velocity, unique (x_vel,y_vel)
counts per timestep, overall unique velocity vectors, and whether any
timestep collapsed to a single unique velocity vector.

Usage:
  python tools/summary_headless_trace.py <trace.csv>

Output: <trace> -> <trace>_summary.json (same folder)
"""
import sys
import os
import json
import math

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def summarize(csv_path):
    try:
        import pandas as pd
    except Exception as e:
        print('pandas required to run this script:', e)
        return 2

    df = pd.read_csv(csv_path)
    # ensure expected columns exist
    for c in ('timestep', 'agent', 'x_vel', 'y_vel'):
        if c not in df.columns:
            print('Missing expected column in CSV:', c)
            return 2

    # per-timestep stats
    grp = df.groupby('timestep')
    per_ts = []
    for ts, g in grp:
        x_mean = float(g['x_vel'].mean()) if not g['x_vel'].isnull().all() else None
        x_std = float(g['x_vel'].std(ddof=0)) if not g['x_vel'].isnull().all() else None
        y_mean = float(g['y_vel'].mean()) if not g['y_vel'].isnull().all() else None
        y_std = float(g['y_vel'].std(ddof=0)) if not g['y_vel'].isnull().all() else None
        unique_pairs = g[['x_vel', 'y_vel']].drop_duplicates().shape[0]
        per_ts.append({'timestep': int(ts), 'x_vel_mean': x_mean, 'x_vel_std': x_std, 'y_vel_mean': y_mean, 'y_vel_std': y_std, 'unique_velocity_pairs': int(unique_pairs)})

    overall_unique = int(df[['x_vel', 'y_vel']].drop_duplicates().shape[0])
    any_single = any(p['unique_velocity_pairs'] == 1 for p in per_ts)

    summary = {
        'csv_path': csv_path,
        'timesteps': int(df['timestep'].max()) + 1,
        'num_agents': int(df['agent'].nunique()),
        'overall_unique_velocity_vectors': overall_unique,
        'any_timestep_single_unique_velocity': bool(any_single),
        'per_timestep_summary_sample': per_ts[:10],
    }

    out_path = csv_path.replace('_trace.csv', '_summary.json')
    if out_path == csv_path:
        out_path = csv_path + '_summary.json'

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    print('Wrote summary to', out_path)
    print('Timesteps:', summary['timesteps'], 'Agents:', summary['num_agents'])
    print('Overall unique velocity vectors:', summary['overall_unique_velocity_vectors'])
    print('Any timestep with single unique velocity vector?', summary['any_timestep_single_unique_velocity'])

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/summary_headless_trace.py <trace.csv>')
        sys.exit(1)
    rc = summarize(sys.argv[1])
    sys.exit(rc)
