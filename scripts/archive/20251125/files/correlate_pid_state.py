"""Correlate PID trace with sim state: print rows where |err_deg| > threshold.

Usage: python scripts/correlate_pid_state.py [path_to_csv] [threshold_deg]
"""
import sys
import os
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'diag_outputs', 'pid_trace_orig_wind.csv')
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0

if not os.path.exists(path):
    print(f"CSV not found: {path}")
    sys.exit(2)

print(f"Loading PID trace: {path}")
# Manual, tolerant parser to handle mixed-row files (older rows with 10 cols, newer with 15)
rows = []
with open(path, 'r', encoding='utf-8') as fh:
    lines = [ln.strip() for ln in fh.readlines() if ln.strip()]

    # expected column sets
    cols10 = ['t','agent','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']
    cols15 = cols10 + ['psi_deg','hd_cmd_deg','r_meas_deg','x_m','y_m']

    # skip header if it matches cols10 or cols15
    start_idx = 0
    first = lines[0]
    if first.split(',')[0].strip().lower() == 't':
        # header present
        start_idx = 1

    for ln in lines[start_idx:]:
        parts = [p for p in ln.split(',')]
        if len(parts) == 10:
            vals = parts
            row = dict(zip(cols10, vals))
        elif len(parts) >= 15:
            # allow extra columns beyond 15 by taking first 15
            vals = parts[:15]
            row = dict(zip(cols15, vals))
        else:
            # malformed/partial line: skip
            continue
        rows.append(row)

    if not rows:
        print('No data rows parsed from CSV')
        sys.exit(4)

    df = pd.DataFrame(rows)
    # coerce numeric columns
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass

if 'err_deg' not in df.columns:
    print('err_deg column not present in CSV; columns:', df.columns.tolist())
    sys.exit(4)

mask = df['err_deg'].abs() >= thr
cnt = mask.sum()
print(f"Found {cnt} rows with |err_deg| >= {thr}° out of {len(df)} total rows")
if cnt == 0:
    # print a small summary
    print(df.head())
    sys.exit(0)

cols_to_show = ['t','agent','err_deg','raw_deg','rud_deg']
# optional columns if present
for c in ['psi_deg','hd_cmd_deg','r_meas_deg','x_m','y_m']:
    if c in df.columns:
        cols_to_show.append(c)

print('\nShowing up to 20 matching rows:')
print(df.loc[mask, cols_to_show].head(20).to_string(index=False))

# Print first instance in full
first_idx = df.loc[mask].index[0]
print('\nFirst matching row (full):')
print(df.loc[first_idx].to_string())
