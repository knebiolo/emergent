"""
Single-command script to run a live salmon ABM simulation with realtime viewer.

Usage:
    python tools/run_live_viewer.py
    python tools/run_live_viewer.py --nagents 5000 --nsteps 1000
"""
import os
import sys
import argparse
import threading
import time
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import numpy as np


def discover_env_files(base_dir):
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description='Run live salmon ABM with viewer')
    parser.add_argument('--nagents', type=int, default=10000, help='Number of agents')
    parser.add_argument('--nsteps', type=int, default=2000, help='Number of timesteps')
    parser.add_argument('--fps', type=float, default=30.0, help='Target frames per second')
    args = parser.parse_args()

    # Setup paths
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
    base = os.path.normpath(base)
    env_files = discover_env_files(base)
    start_poly = os.path.join(base, 'start_loc_river_right.shp')
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'live_viewer'))
    os.makedirs(outdir, exist_ok=True)

    # Create simulation
    print(f"Creating live simulation: {args.nagents} agents, {args.nsteps} steps")
    sim = simulation(
        model_dir=outdir,
        model_name='live_sim',
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, 'longitudinal.shp'),
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=os.path.join(outdir, 'live_sim.h5')
    )

    # Import environment rasters into HDF5
    try:
        h5 = hdf5_io.get_hdf5_obj(sim)
        for ef in env_files:
            try:
                arr, transform, crs = io.enviro_import(ef)
                key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
                hdf5_io.write_dataset(h5, key, np.array(arr))
                if os.path.basename(ef).startswith('depth'):
                    try:
                        t = transform
                        sim.depth_rast_transform = (t.a, t.b, t.c, t.d, t.e, t.f)
                    except Exception:
                        pass
            except Exception as e:
                print(f'Failed to import {ef}: {e}')
    except Exception as e:
        print(f'Environment import error: {e}')

    # Launch viewer in separate thread
    def run_viewer():
        time.sleep(2)  # Give simulation time to start
        from emergent.salmon_abm.realtime_viewer import main as viewer_main
        sys.argv = ['realtime_viewer', '--live', '--port', '9001']
        viewer_main()

    viewer_thread = threading.Thread(target=run_viewer, daemon=True)
    viewer_thread.start()

    # Run simulation with viewer streaming enabled
    print("Starting simulation...")
    sim.run(n=args.nsteps, dt=1.0, return_status=True, 
            viewer_live=True, viewer_host='127.0.0.1', 
            viewer_port=9001, viewer_fps=args.fps)
    print("Simulation complete!")
    sim.close()


if __name__ == '__main__':
    main()
