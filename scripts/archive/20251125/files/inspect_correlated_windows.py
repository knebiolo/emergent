"""Print a few correlated windows (t +/- window_s) for a PID CSV and simhistory NPZ.
Usage: python scripts/inspect_correlated_windows.py pid_csv simhist_npz [window_s]
Defaults: pid_trace_repro_t478.5_correlated.csv and simhist_repro_t478.5.npz
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

args = sys.argv[1:]
pid_csv = args[0] if len(args) > 0 else 'scripts/pid_trace_repro_t478.5_correlated.csv'
simnpz = args[1] if len(args) > 1 else 'scripts/simhist_repro_t478.5.npz'
window_s = float(args[2]) if len(args) > 2 else 5.0

p_pid = Path(pid_csv)
p_npz = Path(simnpz)
if not p_pid.exists() or not p_npz.exists():
    print('Missing files:', p_pid.exists(), p_npz.exists())
    sys.exit(1)

df = pd.read_csv(p_pid)
# choose top 3 largest absolute errors
if 'err_deg' in df.columns:
    df_sorted = df.reindex(df['err_deg'].abs().sort_values(ascending=False).index)
else:
    df_sorted = df

# load simhistory
npz = np.load(p_npz)
t = npz['t']
psi = npz['psi']
hd = npz['hd_cmd']
# pos maybe present
pos = npz['pos'] if 'pos' in npz else None

print(f"Inspecting {pid_csv} matched to {simnpz} -> showing top 3 events with window +/-{window_s}s")
for i, row in df_sorted.head(3).iterrows():
    event_t = float(row['t'])
    print('\n=== Event index', i, 't=', event_t, 'err_deg=', row.get('err_deg'))
    # find nearest sim index
    idx = int(np.argmin(np.abs(t - event_t)))
    lo = max(0, idx - int(window_s / (t[1]-t[0]) if len(t)>1 else 10))
    hi = min(len(t)-1, idx + int(window_s / (t[1]-t[0]) if len(t)>1 else 10))
    print('sim idx range:', lo, hi, 'sim t range:', f"{t[lo]:.1f}", '-', f"{t[hi]:.1f}")
    print('t, psi_deg, hd_cmd_deg')
    for k in range(lo, hi+1):
        print(f"{t[k]:.1f}, {np.degrees(psi[k]):+.3f}, {np.degrees(hd[k]):+.3f}")
    # print surrounding PID rows (approx +/- window)
    df_window = df[(df['t'] >= t[lo]) & (df['t'] <= t[hi])]
    print('\nPID rows in window (t, err_deg, raw_deg, rud_deg, psi_deg, hd_cmd_deg):')
    cols = ['t','err_deg','raw_deg','rud_deg','psi_deg','hd_cmd_deg']
    for _, r in df_window[cols].iterrows():
        print(','.join([str(round(float(r[c]),3)) for c in cols]))

print('\nDone')
