"""Headless debug runner: initialize simulation exactly like run_hecras_opengl.py
but do not start OpenGL. Runs a small number of timesteps and prints relevant debug lines.
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.emergent.salmon_abm.sockeye_SoA_OpenGL import simulation
from src.emergent.salmon_abm.sockeye_dynamic_environment import HECRAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps','-t',type=int,default=30)
    parser.add_argument('--agents','-a',type=int,default=200)
    parser.add_argument('--hecras-plan','-p',type=str,default=None)
    parser.add_argument('--fish-length',type=int,default=500)
    args = parser.parse_args()

    hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
    if args.hecras_plan is None:
        for f in os.listdir(hecras_folder):
            if f.endswith('.p05.hdf'):
                args.hecras_plan = os.path.join(hecras_folder, f)
                break
    if not args.hecras_plan or not os.path.exists(args.hecras_plan):
        print('HECRAS plan not found')
        return 1

    out_dir = os.path.join(REPO_ROOT, 'outputs', 'hecras_run')
    os.makedirs(out_dir, exist_ok=True)

    # Use fast rasterization as in run_hecras_opengl
    env_files = HECRAS.prepare_hecras_rasters(args.hecras_plan, out_dir, resolution=1.0) if False else None

    # fallback: reuse the run_hecras_opengl logic to create rasters by invoking its branch
    # but to keep this quick, we'll call the original tool to produce rasters if they don't exist
    distance_file = os.path.join(out_dir, 'distance_to.tif')
    if not os.path.exists(distance_file):
        print('Creating rasters via fast KDTree (this may take a moment)...')
        # Reuse functionality from run_hecras_opengl.py by importing its main helper code
        import runpy
        runpy.run_path(os.path.join(REPO_ROOT, 'tools', 'run_hecras_opengl.py'), run_name='__main__')
        # After this run, rasters are expected to be in out_dir

    env_files = {
        'elev': os.path.join(out_dir, 'elev.tif'),
        'depth': os.path.join(out_dir, 'depth.tif'),
        'wetted': os.path.join(out_dir, 'wetted.tif'),
        'distance_to': os.path.join(out_dir, 'distance_to.tif'),
        'x_vel': os.path.join(out_dir, 'x_vel.tif'),
        'y_vel': os.path.join(out_dir, 'y_vel.tif'),
        'vel_dir': os.path.join(out_dir, 'vel_dir.tif'),
        'vel_mag': os.path.join(out_dir, 'vel_mag.tif')
    }

    start_poly = os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location', 'start_loc_river_right.shp')

    import tempfile
    temp_model_dir = tempfile.mkdtemp(prefix='emergent_debug_')
    config = {
        'model_dir': temp_model_dir,
        'model_name': 'hecras_run',
        'crs': 'EPSG:26904',
        'basin': 'Nushagak River',
        'water_temp': 10.0,
        'start_polygon': start_poly,
        'env_files': env_files,
        'longitudinal_profile': os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'Longitudinal', 'longitudinal.shp'),
        'fish_length': args.fish_length,
        'num_timesteps': args.timesteps,
        'num_agents': args.agents,
        'use_gpu': False,
        'defer_hdf': False,
    }

    print(f'Initializing simulation with {args.agents} agents...')
    sim = simulation(**config)

    # Load HECRAS nodes and apply mapping like run_hecras_opengl
    import h5py
    hdf_path = os.path.splitext(args.hecras_plan)[0] + '.hdf'
    print('Loading HECRAS nodes for node-based mapping (fast)')
    hdf = h5py.File(hdf_path, 'r')
    try:
        pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
    except Exception:
        pts = None
    node_fields = {}
    try:
        node_fields['depth'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Water Surface'][-1] - np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'))
    except Exception:
        pass
    try:
        node_fields['vel_x'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][-1]
        node_fields['vel_y'] = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity Y'][-1]
    except Exception:
        pass
    hdf.close()

    if pts is not None and node_fields:
        n = sim.num_agents
        if not hasattr(sim, 'depth'):
            sim.depth = np.zeros(n, dtype=float)
        if not hasattr(sim, 'x_vel'):
            sim.x_vel = np.zeros(n, dtype=float)
        if not hasattr(sim, 'y_vel'):
            sim.y_vel = np.zeros(n, dtype=float)
        if not hasattr(sim, 'vel_mag'):
            sim.vel_mag = np.zeros(n, dtype=float)
        if not hasattr(sim, 'wet'):
            sim.wet = np.ones(n, dtype=float)
        if not hasattr(sim, 'distance_to'):
            sim.distance_to = np.zeros(n, dtype=float)

        sim.enable_hecras(pts, node_fields, k=1)
        if 'depth' in node_fields:
            sim.depth = sim.apply_hecras_mapping(node_fields['depth'])
        if 'vel_x' in node_fields:
            sim.x_vel = sim.apply_hecras_mapping(node_fields['vel_x'])
        if 'vel_y' in node_fields:
            sim.y_vel = sim.apply_hecras_mapping(node_fields['vel_y'])
            sim.vel_mag = np.sqrt(sim.x_vel**2 + sim.y_vel**2)

        # Sample distance_to raster at HECRAS node locations
        try:
            import rasterio
            with rasterio.open(env_files['distance_to']) as src:
                coords = [(float(p[0]), float(p[1])) for p in pts]
                samples = np.fromiter((s[0] for s in src.sample(coords)), dtype=float)
                node_fields['distance_to'] = samples
                print(f"Sampled distance_to at {len(samples)} HECRAS nodes (range: {samples.min():.2f}-{samples.max():.2f}m)")
        except Exception as e:
            print(f"Warning: Failed to sample distance_to: {e}")

        if 'distance_to' in node_fields:
            sim.distance_to = sim.apply_hecras_mapping(node_fields['distance_to'])
            print(f"Applied distance_to to agents (range: {sim.distance_to.min():.2f}-{sim.distance_to.max():.2f}m)")

    # Setup PID controller
    try:
        from src.emergent.salmon_abm.sockeye_SoA import PID_controller
        pid = PID_controller(args.agents, k_p=1.0, k_i=0.0, k_d=0.0)
        try:
            pid.interp_PID()
        except Exception:
            pass
    except Exception:
        pid = None

    # Run a short number of timesteps headless to get debug prints
    for t in range(min(10, args.timesteps)):
        sim.timestep(t, 1.0, 9.81, pid)
        # Print any border_cue diagnostic messages are already printed inside border_cue

    print('Headless debug run complete')
    return 0

if __name__ == '__main__':
    sys.exit(main())
