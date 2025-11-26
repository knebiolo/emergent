"""Wait for zigzag_refinement_summary.csv then summarize top candidates and artifacts.

Usage: python wait_for_refine_and_summarize.py

This script polls the repo root for the refinement CSV and, once present,
prints a short report (top candidates by nominal_composite), and lists
key output files (CSV summaries and PNGs) so you can inspect them later.
"""
import time
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_CSV = ROOT / 'zigzag_refinement_summary.csv'
SEARCH_CSV = ROOT / 'zigzag_search_summary.csv'
FIGS_ZIG = ROOT / 'figs' / 'zigzag'
FIGS_SEARCH = ROOT / 'figs' / 'search'

print('Waiting for', REF_CSV)
# wait indefinitely (user requested patience). Poll every 10s.
while True:
    if REF_CSV.exists():
        break
    time.sleep(10)

print('\nREFINEMENT COMPLETE — reading summary...')
try:
    df = pd.read_csv(REF_CSV)
    if 'nominal_composite' in df.columns:
        col = 'nominal_composite'
    elif 'composite_cost' in df.columns:
        col = 'composite_cost'
    else:
        col = None

    if col is not None:
        df_sorted = df.sort_values(col, na_position='last')
    else:
        df_sorted = df

    print('\nTop 5 refinement rows by', col if col else 'order in file')
    cols_to_show = [c for c in ['parent_idx','Kp','Ki','Kf','max_rudder_deg','wind_speed','nominal_rmse','nominal_composite','gust_mean_rmse','gust_std_rmse','gust_worst_rmse','traj_csv'] if c in df_sorted.columns]
    print(df_sorted[cols_to_show].head(5).to_string(index=False))
except Exception as e:
    print('Failed to read or summarize refinement CSV:', e)

# list created artifacts
print('\nArtifacts:')
print(' - search summary:', SEARCH_CSV if SEARCH_CSV.exists() else '(missing)')
print(' - refinement summary:', REF_CSV)

if FIGS_ZIG.exists():
    print('\nPNG files in', FIGS_ZIG)
    for p in sorted(FIGS_ZIG.glob('*.png')):
        print('  -', p)
else:
    print('\nNo figs/zigzag directory found yet.')

if FIGS_SEARCH.exists():
    print('\nPNG files in', FIGS_SEARCH)
    for p in sorted(FIGS_SEARCH.glob('*.png')):
        print('  -', p)
else:
    print('\nNo figs/search directory found yet.')

print('\nDone.')
