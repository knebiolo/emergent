"""
Extract 0-5s snippets for top N runs by event count and plot key columns.

Writes CSV snippets and PNGs under `sweep_results/instrumentation/plots/`.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / 'sweep_results'
INS = SR / 'instrumentation'
PLOTS = INS / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(INS / 'instrumentation_summary.csv')
# pick top 3 runs by n_events (exclude zero-event rows)
summary = summary[summary['n_events']>0]
top = summary.sort_values('n_events', ascending=False).head(3)

def find_trace(runname):
    # candidate trace file is runname + .csv in sweep_results
    cand = SR / (runname + '.csv')
    if cand.exists():
        return cand
    # fallback: find first file whose stem startswith runname
    for p in SR.glob(runname+'*.csv'):
        return p
    return None

for _, row in top.iterrows():
    run = row['run']
    trace = find_trace(run)
    if trace is None:
        print('missing trace for', run)
        continue
    df = pd.read_csv(trace)
    # locate time column
    tcol = None
    for c in ['time','t','timestamp','secs']:
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        tcol = df.columns[0]
    times = pd.to_numeric(df[tcol], errors='coerce')
    mask = times <= times.iloc[0] + 5.0
    df_win = df[mask]
    out_csv = INS / f"{run}_early5s.csv"
    df_win.to_csv(out_csv, index=False)

    # find plotting columns
    def find_col(poss):
        for p in poss:
            if p in df_win.columns:
                return p
        return None

    psi_col = find_col(['psi_deg','psi','heading_deg'])
    hd_col  = find_col(['hd_cmd_deg','hd_cmd'])
    err_col = find_col(['err_deg','err'])
    rud_col = find_col(['rud_deg','rud','rudder_deg','rudder'])

    t = pd.to_numeric(df_win[tcol], errors='coerce').values
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    if psi_col:
        ax.plot(t, pd.to_numeric(df_win[psi_col], errors='coerce'), label='psi_deg')
    if hd_col:
        ax.plot(t, pd.to_numeric(df_win[hd_col], errors='coerce'), label='hd_cmd_deg')
    if err_col:
        ax.plot(t, pd.to_numeric(df_win[err_col], errors='coerce'), label='err_deg')
    if rud_col:
        ax.plot(t, pd.to_numeric(df_win[rud_col], errors='coerce'), label='rud_deg')
    ax.set_title(run)
    ax.set_xlabel('time [s]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"{run}_early5s.png")
    plt.close()
    print('wrote', out_csv, 'and plot')

print('done')
