"""
Headless A/B sweep for zig-zag tests.

Runs short headless zig-zag tests varying wind_force_scale and Kd. Produces per-run
PID trace CSVs and a summary CSV with zig-zag metrics for each combination.

Usage: python scripts/zigzag_ab_sweep.py --T 60
"""
import argparse
import csv
import os
import itertools
import numpy as np

from emergent.ship_abm import config
from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics
from emergent.ship_abm.config import PID_TRACE

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=float, default=60.0)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--out', type=str, default='zigzag_ab_summary.csv')
parser.add_argument('--port', type=str, default='Baltimore')
parser.add_argument('--zigdeg', type=float, default=10.0)
parser.add_argument('--hold', type=float, default=15.0)
parser.add_argument('--wscales', type=float, nargs='*', default=[1.0, 0.35])
parser.add_argument('--kds', type=float, nargs='*', default=[6.0, 4.0, 2.0])
parser.add_argument('--tol', type=float, default=5.0)
args = parser.parse_args()

combos = list(itertools.product(args.wscales, args.kds))
out_rows = []

print(f"Running zig-zag A/B sweep: {len(combos)} runs; T={args.T}s, dt={args.dt}s")

for wscale, kd in combos:
    print(f"\n=== RUN: wind_scale={wscale}, Kd={kd} ===")
    # set global config knobs before simulation instantiation
    config.SHIP_AERO_DEFAULTS['wind_force_scale'] = float(wscale)
    config.CONTROLLER_GAINS['Kd'] = float(kd)

    # per-run PID trace path
    trace_name = f"pid_trace_zigzag_w{wscale}_kd{kd}.csv".replace('.', 'p')
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = os.path.abspath(trace_name)

    # create and run simulation
    sim = simulation(port_name=args.port, dt=args.dt, T=args.T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=args.zigdeg, zigzag_hold=args.hold)
    sim.spawn()
    sim.run()

    # collect metrics
    t = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd = np.array(sim.hd_cmd_history)
    metrics = compute_zigzag_metrics(t, psi, hd, tol=args.tol)

    row = {
        'wind_scale': wscale,
        'Kd': kd,
        'trace': PID_TRACE['path'],
        'peak_overshoot_deg': metrics.get('peak_overshoot_deg', np.nan),
        'settling_time_s': metrics.get('settling_time_s', np.nan),
        'steady_state_error_deg': metrics.get('steady_state_error_deg', np.nan),
        'oscillation_period_s': metrics.get('oscillation_period_s', np.nan)
    }
    out_rows.append(row)
    print('Metrics:', row)

# write summary CSV
out_path = os.path.abspath(args.out)
with open(out_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)

print(f"\nFinished sweep. Summary saved to {out_path}")
