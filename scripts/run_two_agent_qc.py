"""
Simple headless QC run with 2 agents to exercise the simulation-level controller and
produce a PID trace CSV for offline tuning.
"""
# small headless QC runner with simple CLI options
import os
import argparse
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

# parse args
parser = argparse.ArgumentParser(description='Headless two-agent QC run (produces PID trace CSV)')
parser.add_argument('--T', type=float, default=120.0, help='Total simulation time in seconds')
parser.add_argument('--dt', type=float, default=0.5, help='Simulation time step (s)')
parser.add_argument('--trace', type=str, default='pid_trace_two_agent_qc.csv', help='Output PID trace CSV path')
parser.add_argument('--n_agents', type=int, default=2, help='Number of agents to spawn')
parser.add_argument('--port', type=str, default='Baltimore', help='Port name from config to use for the domain')
parser.add_argument('--load-enc', action='store_true', help='If set, load ENC tiles (may be slower)')
args = parser.parse_args()

# configure trace
out_csv = os.path.abspath(args.trace)
PID_TRACE['enabled'] = True
PID_TRACE['path'] = out_csv

# create sim: choose a small domain port (Baltimore exists in config)
sim = simulation(port_name=args.port, dt=args.dt, T=args.T, n_agents=args.n_agents, load_enc=args.load_enc, test_mode=None)

# simple opposing routes: agent0 left->right, agent1 right->left
xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
mid_y = (ymin + ymax) / 2.0
left = xmin + 0.1 * (xmax - xmin)
right = xmax - 0.1 * (xmax - xmin)

sim.waypoints = [
    [np.array([left, mid_y]), np.array([right, mid_y])],
    [np.array([right, mid_y*0.99]), np.array([left, mid_y*0.99])]
]

# spawn agents
sim.spawn()

# run headless
print(f"Running QC sim for {sim.steps} steps (t={sim.steps*sim.dt}s). PID trace -> {out_csv}")
sim.run()

# summary
print(f"Finished. Collisions: {len(sim.collision_events)} Allisions: {len(sim.allision_events)}")
print(f"Trace file: {out_csv}")
