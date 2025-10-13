"""
Simple headless QC run with 2 agents to exercise the simulation-level controller and
produce a PID trace CSV for offline tuning.
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

# configure trace
out_csv = os.path.abspath('pid_trace_two_agent_qc.csv')
PID_TRACE['enabled'] = True
PID_TRACE['path'] = out_csv

# create sim: choose a small domain port (Baltimore exists in config)
sim = simulation(port_name='Baltimore', dt=0.5, T=120.0, n_agents=2, load_enc=False, test_mode=None)

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
