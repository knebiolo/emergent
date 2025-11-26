"""Summarize correlated PID events CSV and print quick diagnostics.

Usage: python scripts/summarize_correlated.py <correlated_csv> [arrival_time]
"""
import sys
import pandas as pd
import numpy as np

path = sys.argv[1] if len(sys.argv)>1 else 'scripts/pid_trace_waypoint_crosswind_correlated_15deg.csv'
arrival = float(sys.argv[2]) if len(sys.argv)>2 else None

df = pd.read_csv(path)
N = len(df)
print(f'Correlated events file: {path}  rows={N}')
if arrival is not None:
    pre = df[df['t'] <= arrival]
    post = df[df['t'] > arrival]
    print(f' events before arrival_time={arrival}: {len(pre)}  after: {len(post)}')
else:
    pre = df; post = pd.DataFrame()

# fraction where |rud_deg| near saturation: use max_rudder from config if present; else assume 20 deg
try:
    from emergent.ship_abm.config import SHIP_PHYSICS
    max_rud = np.degrees(float(SHIP_PHYSICS.get('max_rudder', np.radians(20))))
except Exception:
    max_rud = 20.0

sat_mask = df['rud_deg'].abs() >= (0.95*max_rud)
sat_frac = sat_mask.sum()/N if N>0 else 0.0
print(f' fraction events with rudder saturated (~95% of max {max_rud:.1f}°): {sat_frac:.3f} ({sat_mask.sum()}/{N})')

# detect possible auto-flip windows where raw_cmd large but r_meas opposite sign
if 'r_meas_deg_s' in df.columns and 'raw_deg' in df.columns:
    flip_mask = (df['raw_deg'].abs() > 5.0) & (np.sign(df['raw_deg']) * np.sign(df['r_meas_deg_s']) < 0) & (~df['r_meas_deg_s'].isna())
    print(f' possible auto-flip candidates (raw large but r_meas opp sign): {flip_mask.sum()}')

# basic psi vs hd_cmd stats
if 'psi_deg' in df.columns and 'hd_cmd_deg' in df.columns:
    df['psi_minus_hd'] = df['psi_deg'] - df['hd_cmd_deg']
    df['psi_minus_hd_wrapped'] = ((df['psi_minus_hd'] + 180) % 360) - 180
    med = df['psi_minus_hd_wrapped'].median()
    iqr = df['psi_minus_hd_wrapped'].quantile(0.75) - df['psi_minus_hd_wrapped'].quantile(0.25)
    print(f' psi-hd_cmd wrapped median={med:.2f}°  IQR={iqr:.2f}°')

out_summary = path.replace('.csv','') + '_summary.txt'
with open(out_summary,'w') as fh:
    fh.write(f'file={path}\nrows={N}\nmax_rud={max_rud}\nsat_events={sat_mask.sum()}\nflip_candidates={int(flip_mask.sum()) if "flip_mask" in locals() else 0}\nmedian_psi_minus_hd={med if "med" in locals() else "N/A"}\n')
print(' wrote summary to', out_summary)
