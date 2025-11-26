"""
Headless single-agent Rosario Strait run for candidate tuning verification.
Writes PID trace to scripts/pid_trace_candidate_config.csv for offline analysis.
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE


def main():
    out_csv = os.path.abspath(os.path.join('scripts', 'pid_trace_candidate_config.csv'))
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_csv

    # 60s run with dt matching tests (0.5s used in other rosario scripts)
    sim = simulation(port_name='Rosario Strait', dt=0.5, T=60.0, n_agents=1, load_enc=False)

    # Simple straight across the channel
    xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
    mid_y = (ymin + ymax) / 2.0
    left = xmin + 0.1 * (xmax - xmin)
    right = xmax - 0.1 * (xmax - xmin)

    sim.waypoints = [ [np.array([left, mid_y]), np.array([right, mid_y])] ]

    sim.spawn()
    print(f"Running candidate Rosario sim for {sim.steps} steps (t={sim.steps*sim.dt}s). PID trace -> {out_csv}")
    sim.run()
    print('Finished. Collisions: ', len(sim.collision_events))
    print('Trace file:', out_csv)


if __name__ == '__main__':
    main()
