import sys
import os
from pathlib import Path
import cProfile
import pstats

# Ensure src is on sys.path
root = Path(__file__).resolve().parents[1]
src = root / 'src'
sys.path.insert(0, str(src))

# Minimal profiling run for sockeye
from emergent.salmon_abm.sockeye_SoA import simulation

# Create a minimal environment mapping. Update these paths if different in your workspace.
model_dir = os.path.join(str(root), 'outputs', 'dali_scenario')
# Try to find tif files in model_dir
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

# Prepend model_dir to env_files
env_files = {k: os.path.join(model_dir, v) for k, v in env_files.items()}

# start polygon: try to find a polygon in scenarios or data
start_polygon = None
# fallback: use any shapefile in data/
for p in (root / 'data').rglob('*.shp'):
    start_polygon = str(p)
    break

if start_polygon is None:
    # create a temporary GeoJSON polygon file for starting positions
    import json
    temp = root / 'tools' / 'temp_start_poly.geojson'
    geom = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}
        ]
    }
    with open(temp, 'w') as f:
        json.dump(geom, f)
    start_polygon = str(temp)

# instantiate a tiny simulation
sim = simulation(model_dir=str(model_dir), model_name='prof_test', crs='EPSG:4326', basin='Nushagak River',
                 water_temp=10.0, start_polygon=start_polygon, env_files=env_files,
                 longitudinal_profile=start_polygon, fish_length=500, num_timesteps=50, num_agents=5, use_gpu=False)

# Profile a short run
pr = cProfile.Profile()
pr.enable()
try:
    sim.run('prof_test', n=5, dt=1.0, video=False)
except Exception as e:
    print('Simulation run raised exception:', e)
pr.disable()

# Write stats
stats_file = os.path.join(str(root), 'tools', 'profile_stats.prof')
pr.dump_stats(stats_file)

p = pstats.Stats(pr).strip_dirs().sort_stats('cumtime')
p.print_stats(30)
print('Profile written to', stats_file)
