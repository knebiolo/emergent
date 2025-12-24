"""Analyze headless trace CSV and produce a short JSON summary.

Usage:
  python tools/analyze_headless_trace.py outputs/diagnostics/quick_headless_test_trace.csv
"""
import sys
import json
import pandas as pd


def analyze(fn):
    df = pd.read_csv(fn)
    grp = df.groupby('timestep')
    stats = grp[['x_vel', 'y_vel']].agg(['mean', 'std']).reset_index()
    unique_counts = grp.apply(lambda g: g[['x_vel', 'y_vel']].drop_duplicates().shape[0]).reset_index()
    unique_counts.columns = ['timestep', 'unique_vel_pairs']

    merged = stats.merge(unique_counts, on='timestep')
    overall_unique = df[['x_vel', 'y_vel']].drop_duplicates().shape[0]
    any_single = (merged['unique_vel_pairs'] == 1).any()

    summary = {
        'timesteps': int(df['timestep'].max()) + 1,
        'num_agents': int(df['agent'].nunique()),
        'overall_unique_velocity_vectors': int(overall_unique),
        'any_timestep_single_unique_velocity': bool(any_single),
        'per_timestep_sample': merged.head(10).to_dict(orient='records')
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/analyze_headless_trace.py <trace.csv>')
        sys.exit(1)
    analyze(sys.argv[1])
