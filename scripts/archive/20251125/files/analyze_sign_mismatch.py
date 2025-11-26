"""
Detect rows in given pid_trace CSVs where sign(raw_deg) != sign(P_deg) for large magnitudes.
Print first occurrence and ±1s context.
"""
import pandas as pd
import glob
import math

candidates = ['scripts/pid_trace_expt_2p0.csv','scripts/pid_trace_expt_3p0.csv']
window_s = 1.0

def first_mismatch(df0):
    for i,row in df0.iterrows():
        p = row['P_deg']
        r = row['raw_deg']
        if abs(p) > 5.0 and abs(r) > 5.0 and ((p>0) != (r>0)):
            return i
    return None

for p in candidates:
    print('\n===', p)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print('failed read', e)
        continue
    df0 = df[df['agent']==0].reset_index(drop=True)
    if df0.empty:
        print('no agent 0')
        continue
    idx = first_mismatch(df0)
    if idx is None:
        print('no large-sign mismatch found')
        continue
    t0 = df0.loc[idx,'t']
    print('first mismatch index', idx, 't=', t0)
    # compute window
    dt = df0.loc[1,'t'] - df0.loc[0,'t'] if len(df0)>1 else 0.1
    win = int(max(1, round(window_s/dt)))
    start = max(0, idx-win)
    end = min(len(df0)-1, idx+win)
    cols = ['t','err_deg','P_deg','D_deg','raw_deg','rud_deg']
    for j in range(start,end+1):
        row = df0.loc[j]
        vals = [f"{row[c]:.3f}" if c in row.index else 'NA' for c in cols]
        print(', '.join(vals))

print('\nDone')
