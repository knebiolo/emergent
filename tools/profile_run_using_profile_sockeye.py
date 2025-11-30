import sys
import os
from pathlib import Path
import argparse
import cProfile
import pstats

# Ensure src is on sys.path
root = Path(__file__).resolve().parents[1]
src = root / 'src'
sys.path.insert(0, str(src))

from emergent.salmon_abm.sockeye_SoA import simulation


def build_sim_and_run(num_agents, n_steps, out_stats, dt=1.0):
    model_dir = os.path.join(str(root), 'outputs', 'dali_scenario')
    env_files = {
        'x_vel': 'vel_x.tif',
        'y_vel': 'vel_y.tif',
        'depth': 'depth.tif',
        'wsel': 'wsel.tif',
        'elev': 'elev.tif',
        'vel_dir': 'vel_dir.tif',
        'vel_mag': 'vel_mag.tif',
        'wetted': 'wetted.tif'
    }
    env_files = {k: os.path.join(model_dir, v) for k, v in env_files.items()}

    # find a start polygon
    start_polygon = None
    for p in (root / 'data').rglob('*.shp'):
        start_polygon = str(p)
        break
    if start_polygon is None:
        # fallback
        start_polygon = os.path.join(str(root), 'tools', 'temp_start_poly.geojson')

    sim = simulation(model_dir=str(model_dir), model_name='prof_test', crs='EPSG:4326', basin='Nushagak River',
                     water_temp=10.0, start_polygon=start_polygon, env_files=env_files,
                     longitudinal_profile=start_polygon, fish_length=500, num_timesteps=n_steps, num_agents=num_agents, use_gpu=False)

    pr = cProfile.Profile()
    pr.enable()
    try:
        sim.run('prof_test', n=n_steps, dt=dt, video=False)
    except Exception as e:
        # Ensure profile is recorded even if run raises
        print('sim.run raised:', e)
    pr.disable()
    pr.dump_stats(out_stats)

    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=5)
    parser.add_argument('--out', default='profile_run.pstats')
    args = parser.parse_args()
    build_sim_and_run(args.agents, args.timesteps, args.out)


if __name__ == '__main__':
    main()
