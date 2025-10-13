"""
Headless run: Rosario Strait, 2 vessels, PID trace.

Usage: python scripts/run_rosario_two_vessels.py
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE


def main():
    out_csv = os.path.abspath('pid_trace_rosario.csv')
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_csv

    sim = simulation(port_name='Rosario Strait', dt=0.5, T=300.0, n_agents=2, load_enc=False)

    xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
    mid_y = (ymin + ymax) / 2.0
    left = xmin + 0.08 * (xmax - xmin)
    right = xmax - 0.08 * (xmax - xmin)

    sim.waypoints = [
        [np.array([left, mid_y]), np.array([right, mid_y])],
        [np.array([right, mid_y*0.995]), np.array([left, mid_y*0.995])]
    ]

    sim.spawn()
    print(f"Running Rosario sim for {sim.steps} steps (t={sim.steps*sim.dt}s). PID trace -> {out_csv}")
    sim.run()
    print(f"Finished. Collisions: {len(sim.collision_events)} Allisions: {len(sim.allision_events)}")
    print('Trace file:', out_csv)


if __name__ == '__main__':
    main()
