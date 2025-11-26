"""
Compute summary metrics for pid_trace_expt_*.csv created by the experiment driver.
Usage: python scripts\compute_expt_summary.py
"""
import glob
import math
import pandas as pd

paths = sorted(glob.glob('scripts/pid_trace_expt_*.csv'))
if not paths:
    print('No experiment traces found in scripts/')
    raise SystemExit(1)

rows = []
for p in paths:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f'Failed to read {p}: {e}')
        continue
    if 'agent' not in df.columns:
        print(f'{p}: unexpected columns: {df.columns.tolist()}')
        continue
    df0 = df[df['agent'] == 0]
    total = len(df0)
    if total == 0:
        rows.append((p, math.nan, math.nan, math.nan, 0.0))
        continue
    max_err = df0['err_deg'].abs().max()
    max_raw = df0['raw_deg'].abs().max()
    mean_rud = df0['rud_deg'].mean()
    max_rudder_deg = math.degrees(0.3490658503988659) if 'max_rudder' not in () else None
    # saturation threshold read from header if present
    # fallback: 20 deg (0.349 rad)
    try:
        sat_count = (df0['rud_deg'].abs() >= (20.0 - 1e-6)).sum()
    except Exception:
        sat_count = 0
    sat_frac = sat_count / total if total>0 else 0.0
    rows.append((p, max_err, max_raw, mean_rud, sat_frac))

# print human-friendly table
print('\nComputed summary for experiment traces:')
print(f"{'trace':<40} {'max_err_deg':>12} {'max_raw_deg':>12} {'mean_rud_deg':>12} {'sat_frac':>8}")
for p, me, mr, mrud, sf in rows:
    me_s = f"{me:.2f}" if not math.isnan(me) else 'nan'
    mr_s = f"{mr:.2f}" if not math.isnan(mr) else 'nan'
    mrud_s = f"{mrud:.2f}" if not math.isnan(mrud) else 'nan'
    print(f"{p:<40} {me_s:>12} {mr_s:>12} {mrud_s:>12} {sf:8.2%}")

print('\nDone')
