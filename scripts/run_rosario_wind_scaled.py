"""
Run Rosario Strait headless but scale wind magnitude by a factor for testing.

Usage: python scripts/run_rosario_wind_scaled.py --scale 0.5
"""
import os
import argparse
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.ofs_loader import get_wind_fn
from emergent.ship_abm.config import PID_TRACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=0.5, help='Multiply sampled wind by this factor')
    parser.add_argument('--T', type=float, default=300.0, help='Simulation time (s)')
    args = parser.parse_args()

    out_csv = os.path.abspath('pid_trace_rosario_scaled.csv')
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_csv

    sim = simulation(port_name='Rosario Strait', dt=0.5, T=args.T, n_agents=2, load_enc=True)

    # Wrap the sim.wind_fn to scale the output
    base_wind = sim.wind_fn
    def scaled_wind(lon, lat, when):
        out = base_wind(lon, lat, when)
        return out * args.scale
    sim.wind_fn = scaled_wind

    xmin, ymin, xmax, ymax = sim.minx, sim.miny, sim.maxx, sim.maxy
    mid_y = (ymin + ymax) / 2.0
    left = xmin + 0.12 * (xmax - xmin)
    right = xmax - 0.12 * (xmax - xmin)

    sim.waypoints = [
        [np.array([left, mid_y + 30.0]), np.array([right, mid_y + 30.0])],
        [np.array([right, mid_y - 30.0]), np.array([left, mid_y - 30.0])]
    ]

    sim.spawn()
    print(f"Running Rosario scaled-wind sim (scale={args.scale}) -> trace {out_csv}")
    sim.run()
    print(f"Finished. Collisions: {len(sim.collision_events)} Allisions: {len(sim.allision_events)}")
    print('Trace file:', out_csv)


if __name__ == '__main__':
    main()
