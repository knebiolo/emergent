"""Run refinement and plotting using existing zigzag_search_summary.csv

This helper ensures we can invoke the local refinement step from the
`zigzag_search_and_refine.py` driver in a PowerShell-friendly way.
"""
import sys
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / 'scripts' / 'zigzag_search_and_refine.py'
SUMMARY_CSV = ROOT / 'zigzag_search_summary.csv'

if not SUMMARY_CSV.exists():
    print('ERROR: search summary not found at', SUMMARY_CSV)
    sys.exit(2)

spec = importlib.util.spec_from_file_location('zigzag_search_and_refine', str(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print('[REFINE-ONLY] Loaded driver module; reading search summary...')
df = pd.read_csv(SUMMARY_CSV)
print('[REFINE-ONLY] rows=', len(df))

# Run refinement (top 3, 10 gust trials) — adjust as needed
print('[REFINE-ONLY] starting local_refine(top_n=3, gust_trials=10)')
df_ref = mod.local_refine(df, top_n=3, gust_trials=10)
print('[REFINE-ONLY] local_refine completed; wrote zigzag_refinement_summary.csv')

# Make plots
print('[REFINE-ONLY] generating plots')
try:
    mod.make_plots(df, df_ref)
    print('[REFINE-ONLY] plots done')
except Exception as e:
    print('[REFINE-ONLY] plotting failed:', e)

print('DONE')
