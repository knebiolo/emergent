"""Parameter sweep for PID gains.

Runs short headless simulations varying Kp, Ki, Kd (grid) and records RMS heading error
for agent 1 (the moving agent). Saves a CSV summary and a PNG heatmap.

Usage: run from project root: python scripts/parameter_sweep_pid.py
"""
import os
import csv
import itertools
import math
import tempfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE


def run_sim_with_gains(kp, ki, kd, T=20.0, dt=0.5, trace_path=None):
    """Create and run a short two-agent sim with provided PID gains.

    Returns times (numpy array) and heading error for agent 1 as radians.
    """
    # Create simulation using project API (matches run_two_agent_qc.py)
    out_csv = trace_path or os.path.abspath('pid_trace_temp.csv')
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_csv

    sim = simulation(port_name='Baltimore', dt=dt, T=T, n_agents=2, load_enc=False)

    # opposing straight-line waypoints
    xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
    mid_y = (ymin + ymax) / 2.0
    left = xmin + 0.1 * (xmax - xmin)
    right = xmax - 0.1 * (xmax - xmin)

    sim.waypoints = [
        [np.array([left, mid_y]), np.array([right, mid_y])],
        [np.array([right, mid_y*0.99]), np.array([left, mid_y*0.99])]
    ]

    # apply tuning gains into sim.tuning
    sim.tuning['Kp'] = kp
    sim.tuning['Ki'] = ki
    sim.tuning['Kd'] = kd

    sim.spawn()
    print(f'Running short sim T={T}s dt={dt}s -> trace {out_csv}')
    sim.run()
    # read trace if present else compute heading error from sim history
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        df1 = df[df['agent'] == 1]
        errs = np.deg2rad(df1['err_deg'].values)
        t = df1['t'].values
        return t, errs
    else:
        # fallback: try to get heading error from sim.history (if implemented)
        if hasattr(sim, 'history') and sim.history:
            # expect sim.history to contain tuples (t, states) or a DataFrame
            # best-effort extraction
            times = []
            errs = []
            for rec in getattr(sim, 'history'):
                try:
                    times.append(rec['t'])
                    errs.append(rec['err'][1])
                except Exception:
                    pass
            return np.array(times), np.array(errs)
    return np.array([]), np.array([])


def rms(x):
    if len(x) == 0:
        return float('nan')
    return math.sqrt(np.mean(np.square(x)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', help='Output prefix', default='sweep_pid')
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=20.0)
    args = parser.parse_args()

    # small grid for speed
    Kp_vals = [0.2, 0.5, 1.0]
    Ki_vals = [0.0, 0.01, 0.05]
    Kd_vals = [0.0, 0.1, 0.5]

    results = []
    # temp dir for per-run traces
    out_dir = os.path.abspath('.')
    summary_rows = []
    total = len(Kp_vals) * len(Ki_vals) * len(Kd_vals)
    i = 0
    for kp, ki, kd in itertools.product(Kp_vals, Ki_vals, Kd_vals):
        i += 1
        print(f'Run {i}/{total}: Kp={kp} Ki={ki} Kd={kd}')
        trace_name = os.path.join(out_dir, f'trace_kp{kp}_ki{ki}_kd{kd}.csv')
        try:
            t, errs = run_sim_with_gains(kp, ki, kd, T=args.T, dt=args.dt, trace_path=trace_name)
            e = rms(errs)
        except Exception as e:
            print('Run failed:', e)
            e = float('nan')
        summary_rows.append({'Kp': kp, 'Ki': ki, 'Kd': kd, 'rms_err_rad': e})

    df = pd.DataFrame(summary_rows)
    csv_out = args.out + '_summary.csv'
    df.to_csv(csv_out, index=False)
    print('Saved summary CSV to', csv_out)

    # Simple 2D heatmap: fix Ki at middle value, vary Kp vs Kd
    mid_ki = Ki_vals[len(Ki_vals) // 2]
    sub = df[df['Ki'] == mid_ki]
    pivot = sub.pivot(index='Kd', columns='Kp', values='rms_err_rad')
    plt.figure(figsize=(6, 5))
    plt.title(f'RMS heading error (Ki={mid_ki})')
    im = plt.imshow(pivot.values, origin='lower', aspect='auto')
    plt.colorbar(im, label='rms err (rad)')
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(c) for c in pivot.index])
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    png_out = args.out + '_heatmap.png'
    plt.savefig(png_out)
    print('Saved heatmap to', png_out)


if __name__ == '__main__':
    main()
