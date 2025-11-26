"""
Inspect pid_trace_expt_*.csv traces and detect first large spike in err_deg or raw_deg.
Print ±1s around the spike and summarize which PID component grew first.
"""
import glob
import pandas as pd
import math

paths = sorted(glob.glob('scripts/pid_trace_expt_*.csv'))
if not paths:
    print('No experiment traces found')
    raise SystemExit(1)

threshold_err = 20.0  # degrees considered a large heading error
threshold_raw = 15.0  # degrees raw_cmd considered large
window_s = 1.0

for p in paths:
    print('\n--- Inspecting', p)
    df = pd.read_csv(p)
    if 'agent' not in df.columns:
        print('unexpected format, skipping')
        continue
    df0 = df[df['agent'] == 0].reset_index(drop=True)
    if df0.empty:
        print('no agent-0 rows')
        continue
    # find first index where either err or raw exceed thresholds
    idx = None
    for i, row in df0.iterrows():
        try:
            if abs(row['err_deg']) >= threshold_err or abs(row['raw_deg']) >= threshold_raw:
                idx = i
                break
        except Exception:
            continue
    if idx is None:
        print('No large spike found (thresholds err>=%.1f raw>=%.1f)' % (threshold_err, threshold_raw))
        continue
    t0 = df0.loc[idx, 't']
    print('First spike at index', idx, 't=', t0)
    # window rows
    dt = df0.loc[1,'t'] - df0.loc[0,'t'] if len(df0)>1 else 0.1
    win = int(max(1, round(window_s / dt)))
    start = max(0, idx - win)
    end = min(len(df0)-1, idx + win)
    print('\nRows around spike (t, err_deg, r_des_deg, derr_deg, P_deg, I_deg, D_deg, raw_deg, rud_deg):')
    cols = ['t','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']
    for j in range(start, end+1):
        row = df0.loc[j]
        vals = [f"{row[c]:.3f}" if c in row.index else 'NA' for c in cols]
        print(', '.join(vals))
    # quick component that moved most
    comp_changes = {}
    for comp in ['P_deg','I_deg','D_deg','raw_deg']:
        before = df0.loc[start:idx-1, comp].abs().max() if idx-1>=start else 0.0
        at = abs(df0.loc[idx, comp])
        comp_changes[comp] = at - before
    sorted_comp = sorted(comp_changes.items(), key=lambda x: -x[1])
    print('\nComponent delta (at spike minus prior max):')
    for comp, delta in sorted_comp:
        print(f"  {comp}: {delta:.3f}")

print('\nDone')
