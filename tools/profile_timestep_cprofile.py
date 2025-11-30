"""Run a short simulation loop under cProfile and report hotspots.

Usage:
    python tools/profile_timestep_cprofile.py --agents 100 --timesteps 10

This script creates a small simulation configured to use HECRAS per-agent sampling
but without writing full-grid rasters. It profiles the `sim.timestep()` calls and
prints a top-50 report by cumulative time and writes a pstats file `profile.pstats`.
"""
import argparse
import cProfile
import pstats
import io
import os
import sys

# Allow running from repo root when executed from tools/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emergent.salmon_abm.sockeye_SoA import simulation


def build_sim(num_agents, use_hecras=True, hecras_write_rasters=False):
    # Build minimal env_files mapping similar to other profiling tools
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_dir = os.path.join(repo_root, 'outputs', 'dali_scenario')
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

    # find or create a tiny start polygon
    start_polygon = None
    for p in (os.path.join(repo_root, 'data')).replace('\\', '/') and []:
        pass

    # use user's longitudinal shapefile if present
    user_long_dir = r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\Longitudinal"
    tmp = os.path.join(repo_root, 'tools', 'temp_start_poly.geojson')
    line_tmp = os.path.join(repo_root, 'tools', 'temp_longitudinal_line.geojson')
    longitudinal_profile = None
    try:
        if os.path.isdir(user_long_dir):
            # pick the first shapefile in the folder
            for f in os.listdir(user_long_dir):
                if f.lower().endswith('.shp'):
                    longitudinal_profile = os.path.join(user_long_dir, f)
                    break
    except Exception:
        longitudinal_profile = None
    if not os.path.exists(tmp):
        import json
        geom = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}
            ]
        }
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(geom, f)
    start_polygon = tmp
    # create a simple LineString geojson for longitudinal_profile if no user file found
    if longitudinal_profile is None:
        if not os.path.exists(line_tmp):
            import json
            line = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0,0],[1,1],[2,2]]}}
                ]
            }
            with open(line_tmp, 'w') as f:
                json.dump(line, f)
        longitudinal_profile = line_tmp

    # Instantiate simulation with required args
    sim = simulation(model_dir=str(model_dir), model_name='prof_test', crs='EPSG:4326', basin='Nushagak River',
                     water_temp=10.0, start_polygon=start_polygon, env_files=env_files,
                     longitudinal_profile=longitudinal_profile, fish_length=500, num_timesteps=100, num_agents=num_agents, use_gpu=False,
                     hecras_plan_path=None, hecras_fields=None, hecras_k=8, use_hecras=use_hecras, hecras_write_rasters=hecras_write_rasters)
    # Ensure mental_map_transform exists to satisfy geo_to_pixel
    try:
        import rasterio
        from rasterio.transform import Affine
        sim.mental_map_transform = Affine.translation(0, 0)
    except Exception:
        class _AffStub:
            def __invert__(self):
                return self
        sim.mental_map_transform = _AffStub()

    return sim


def profile_run(num_agents, timesteps, out_stats='profile.pstats'):
    sim = build_sim(num_agents)

    pr = cProfile.Profile()
    pr.enable()

    # timestep signature requires (t, dt, g, pid_controller) — pass simple placeholders
    # Create a minimal PID stub if code expects PID controller
    class _PIDStub:
        def __init__(self):
            pass
    pid = _PIDStub()
    for t in range(timesteps):
        sim.timestep(t, 1.0, 9.81, pid)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.dump_stats(out_stats)

    s = io.StringIO()
    ps.stream = s
    ps.print_stats(50)
    print(s.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=100)
    parser.add_argument('--timesteps', type=int, default=10)
    parser.add_argument('--out', default='profile.pstats')
    args = parser.parse_args()

    profile_run(args.agents, args.timesteps, args.out)


if __name__ == '__main__':
    main()
