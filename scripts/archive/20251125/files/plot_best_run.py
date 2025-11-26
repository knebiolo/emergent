"""Create a polished composite figure for the best parameter-sweep run.

Finds the minimum RMS row in `sweep_pid_summary.csv`, loads its trace file
(`trace_kp{Kp}_ki{Ki}_kd{Kd}.csv`) and produces `best_run_summary.png`.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(summary_csv='sweep_pid_summary.csv'):
    df = pd.read_csv(summary_csv)
    best = df.loc[df['rms_err_rad'].idxmin()]
    kp, ki, kd = best['Kp'], best['Ki'], best['Kd']
    trace_fn = f'trace_kp{kp}_ki{ki}_kd{kd}.csv'
    if not os.path.exists(trace_fn):
        print('Trace file not found:', trace_fn)
        return
    tdf = pd.read_csv(trace_fn)
    t = tdf['t']
    agent1 = tdf[tdf['agent'] == 1]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2)

    ax_track = fig.add_subplot(gs[0, :])
    # Attempt to reconstruct approximate track from psi/err (not available) — show rudder/time instead
    ax_err = fig.add_subplot(gs[1, 0])
    ax_pid = fig.add_subplot(gs[1, 1])
    ax_rud = fig.add_subplot(gs[2, :])

    ax_err.plot(agent1['t'], agent1['err_deg'], color='C1')
    ax_err.set_title('Heading error (deg) — agent 1')
    ax_err.set_xlabel('time (s)')

    ax_pid.plot(agent1['t'], agent1['P_deg'], label='P')
    ax_pid.plot(agent1['t'], agent1['I_deg'], label='I')
    ax_pid.plot(agent1['t'], agent1['D_deg'], label='D')
    ax_pid.set_title('PID terms (deg) — agent 1')
    ax_pid.legend()

    ax_rud.plot(agent1['t'], agent1['raw_deg'], label='raw (deg)')
    ax_rud.plot(agent1['t'], agent1['rud_deg'], label='applied rudder (deg)')
    ax_rud.set_title('Rudder command (deg)')
    ax_rud.set_xlabel('time (s)')
    ax_rud.legend()

    fig.suptitle(f'Best run Kp={kp} Ki={ki} Kd={kd} RMS={best["rms_err_rad"]:.4f} rad')
    out = 'best_run_summary.png'
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    print('Saved', out)


if __name__ == '__main__':
    main()
