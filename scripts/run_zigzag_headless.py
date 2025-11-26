"""
Headless zig-zag test runner. Runs the simulation in test_mode='zigzag' without launching the GUI
and writes a PID trace CSV for offline analysis.

Usage: python scripts/run_zigzag_headless.py --T 120 --trace pid_trace_zigzag.csv
"""
import argparse
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics
from emergent.ship_abm.config import PID_TRACE

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=float, default=120.0)
parser.add_argument('--dt', type=float, default=0.5)
parser.add_argument('--trace', type=str, default='pid_trace_zigzag.csv')
parser.add_argument('--port', type=str, default='Baltimore')
parser.add_argument('--zigdeg', type=float, default=10.0)
parser.add_argument('--hold', type=float, default=30.0)
args = parser.parse_args()

# configure PID trace
PID_TRACE['enabled'] = True
PID_TRACE['path'] = os.path.abspath(args.trace)

# create simulation in zigzag test mode
sim = simulation(port_name=args.port, dt=args.dt, T=args.T, n_agents=1, load_enc=False,
                 test_mode='zigzag', zigzag_deg=args.zigdeg, zigzag_hold=args.hold)

print(f"Running headless zigzag for T={args.T}s (dt={args.dt}s). PID trace -> {PID_TRACE['path']}")
sim.spawn()
sim.run()

# collect histories
t_data = np.array(sim.t_history)
actual_heading = np.array(sim.psi_history)
cmd_heading = np.array(sim.hd_cmd_history)

metrics = compute_zigzag_metrics(t_data, actual_heading, cmd_heading, tol=5.0)
print('Zig-zag metrics:', metrics)
print('Trace file:', PID_TRACE['path'])
