"""
Plot comparison between baseline repro trace and a selected sweep trace for a short window.
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path('scripts/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

base = Path('scripts/pid_trace_repro_t478.5.csv')
best = Path('scripts/sweep_traces/pid_b0.08_n70.csv')

if not base.exists() or not best.exists():
    print('Missing one of the traces:', base, best)
    raise SystemExit(1)

df0 = pd.read_csv(base)
df1 = pd.read_csv(best)

# select agent 0 and t window
w0 = (df0['agent'] == 0) & (df0['t'] <= 30)
w1 = (df1['agent'] == 0) & (df1['t'] <= 30)

tag = 'baseline_vs_best_first30s'
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

axs[0].plot(df0.loc[w0,'t'], df0.loc[w0,'err_deg'], label='baseline')
axs[0].plot(df1.loc[w1,'t'], df1.loc[w1,'err_deg'], label='best', linestyle='--')
axs[0].set_ylabel('err_deg')
axs[0].legend()

axs[1].plot(df0.loc[w0,'t'], df0.loc[w0,'raw_deg'], label='baseline')
axs[1].plot(df1.loc[w1,'t'], df1.loc[w1,'raw_deg'], label='best', linestyle='--')
axs[1].set_ylabel('raw_deg')
axs[1].legend()

axs[2].plot(df0.loc[w0,'t'], df0.loc[w0,'rud_deg'], label='baseline')
axs[2].plot(df1.loc[w1,'t'], df1.loc[w1,'rud_deg'], label='best', linestyle='--')
axs[2].set_ylabel('rud_deg')
axs[2].legend()

# psi_deg/hd_cmd_deg might be missing in baseline; guard
if 'psi_deg' in df0.columns:
    axs[3].plot(df0.loc[w0,'t'], df0.loc[w0,'psi_deg'], label='psi baseline')
if 'psi_deg' in df1.columns:
    axs[3].plot(df1.loc[w1,'t'], df1.loc[w1,'psi_deg'], label='psi best', linestyle='--')
axs[3].set_ylabel('psi_deg')
axs[3].legend()

if 'hd_cmd_deg' in df0.columns:
    axs[4].plot(df0.loc[w0,'t'], df0.loc[w0,'hd_cmd_deg'], label='hd_cmd baseline')
if 'hd_cmd_deg' in df1.columns:
    axs[4].plot(df1.loc[w1,'t'], df1.loc[w1,'hd_cmd_deg'], label='hd_cmd best', linestyle='--')
axs[4].set_ylabel('hd_cmd_deg')
axs[4].set_xlabel('t (s)')
axs[4].legend()

fig.tight_layout()
out = OUT_DIR / (tag + '.png')
fig.savefig(out)
print('Saved', out)
