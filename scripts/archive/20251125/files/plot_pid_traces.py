"""
Plot PID traces for long-run and best-case runs.
Generates PNGs in `figs/` and appends short captions to `docs/PID_tuning_readme.md`.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRACES_DIR = 'traces'
FIGS_DIR = 'figs'
os.makedirs(FIGS_DIR, exist_ok=True)

# pick files: the long run and the best Kd=3.0 trace(s)
long_trace = 'pid_trace_zigzag_long.csv'
best_traces = sorted(glob.glob(os.path.join(TRACES_DIR, 'pid_trace_kd3p0_*.csv')))

plots = []

def load_trace(path):
    df = pd.read_csv(path)
    # Current trace columns: t,agent,err_deg,derr_deg,P_deg,I_deg,D_deg,raw_deg,rud_deg
    # Normalize column names for plotting convenience
    colmap = {
        'err_deg': 'psi_err_deg',
        'P_deg': 'P_deg',
        'I_deg': 'I_deg',
        'D_deg': 'D_deg',
        'raw_deg': 'rudder_cmd_deg',
        'rud_deg': 'rudder_applied_deg'
    }
    for old, new in colmap.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df

# helper to produce plots for a trace
def plot_trace(df, tag):
    t = df['t'].values
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    # heading error
    axs[0].plot(t, df['psi_err_deg'], label='heading error (deg)')
    axs[0].axhline(0, color='k', lw=0.5)
    axs[0].legend()
    axs[0].grid(True)

    # P/I/D
    axs[1].plot(t, df['P_deg'], label='P (deg)')
    axs[1].plot(t, df['I_deg'], label='I (deg)')
    axs[1].plot(t, df['D_deg'], label='D (deg)')
    axs[1].legend()
    axs[1].grid(True)

    # commanded vs applied rudder
    axs[2].plot(t, df['rudder_cmd_deg'], label='rudder_cmd (deg)')
    axs[2].plot(t, df['rudder_applied_deg'], label='rudder_applied (deg)', linestyle='--')
    axs[2].legend()
    axs[2].grid(True)

    # zoomed heading error last 120s
    axs[3].plot(t, df['psi_err_deg'], label='heading error (deg)')
    axs[3].set_xlim(max(0, t[-1]-120), t[-1])
    axs[3].legend()
    axs[3].grid(True)

    axs[-1].set_xlabel('time (s)')
    fig.suptitle(f'PID trace: {tag}')
    out = os.path.join(FIGS_DIR, f'pid_trace_{tag}.png')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out)
    plt.close(fig)
    return out

notes = []
# plot long trace if available
if os.path.exists(long_trace):
    df_long = load_trace(long_trace)
    out_long = plot_trace(df_long, 'long')
    notes.append((out_long, 'Long run (360s) with conservative tuning (Kd=3.0)'))

# plot best traces
for p in best_traces:
    df = load_trace(p)
    tag = os.path.splitext(os.path.basename(p))[0]
    out = plot_trace(df, tag)
    notes.append((out, f'Best-case sweep trace: {tag}'))

# append to README
readme = 'docs/PID_tuning_readme.md'
with open(readme, 'a', encoding='utf-8') as f:
    f.write('\n\n## Diagnostic plots\n')
    for img, caption in notes:
        f.write(f'![{caption}]({img})\n\n')
        f.write(f'*{caption}* — see `{img}`.\n\n')

print('Plots generated and appended to', readme)
