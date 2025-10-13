"""
Run a headless Rosario Strait sim with ENC and currents and fixed straight routes for two agents.

Usage: python scripts/run_rosario_two_vessels_fixed_routes.py
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE


def main():
    out_csv = os.path.abspath('pid_trace_rosario_fixed.csv')
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_csv

    # load_enc=True to use ENC + OFS samplers
    sim = simulation(port_name='Rosario Strait', dt=0.5, T=300.0, n_agents=2, load_enc=True)

    # Create corrected straight routes closer to fairway center
    xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
    mid_y = (ymin + ymax) / 2.0
    # move slightly inward from corners
    left = xmin + 0.12 * (xmax - xmin)
    right = xmax - 0.12 * (xmax - xmin)

    # define routes offset in y to avoid exact overlap
    sim.waypoints = [
        [np.array([left, mid_y + 50.0]), np.array([right, mid_y + 50.0])],
        [np.array([right, mid_y - 50.0]), np.array([left, mid_y - 50.0])]
    ]

    sim.spawn()
    print(f"Running Rosario fixed-route sim for {sim.steps} steps (t={sim.steps*sim.dt}s). PID trace -> {out_csv}")
    sim.run()
    print(f"Finished. Collisions: {len(sim.collision_events)} Allisions: {len(sim.allision_events)}")
    print('Trace file:', out_csv)


if __name__ == '__main__':
    main()
