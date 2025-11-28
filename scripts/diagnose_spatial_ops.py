from pathlib import Path
import json
from datetime import datetime

from emergent.ship_abm.simulation_core import simulation

cfg_path = Path('scenarios/dali_bridge_collision_config.json')
raw = cfg_path.read_bytes()
import codecs
if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(b"\xff\xfe"):
    cfg_text = raw.decode('utf-16')
elif raw.startswith(codecs.BOM_UTF16_BE) or raw.startswith(b"\xfe\xff"):
    cfg_text = raw.decode('utf-16')
elif raw.startswith(codecs.BOM_UTF8) or raw.startswith(b"\xef\xbb\xbf"):
    cfg_text = raw.decode('utf-8-sig')
else:
    try:
        cfg_text = raw.decode('utf-8')
    except Exception:
        cfg_text = raw.decode('latin-1')
config = json.loads(cfg_text)
port = config['simulation']['port_name']

sim = simulation(port_name=port, dt=1.0, T=1.0, n_agents=0, load_enc=True)
print('sim.crs_utm =', sim.crs_utm)
enc = getattr(sim, 'enc_data', {})
print('ENC layer keys:', list(enc.keys()))

# Print bounds of bridge and land layers
for k in ('LNDARE', 'COALNE', 'BRIDGE'):
    layer = enc.get(k)
    if layer is None:
        print(k, 'layer missing')
        continue
    if isinstance(layer, list):
        for i, gdf in enumerate(layer):
            try:
                print(f'{k}[{i}] bounds:', gdf.total_bounds)
                print(f'{k}[{i}] crs:', getattr(gdf, 'crs', None))
            except Exception as e:
                print(f'{k}[{i}] read error:', e)
    else:
        try:
            print(f'{k} bounds:', layer.total_bounds)
            print(f'{k} crs:', getattr(layer, 'crs', None))
        except Exception as e:
            print(k, 'error:', e)

# spawn a ship at configured start and get hull
from pyproj import Transformer
waypoints_lonlat = config['route']['waypoints_lonlat']
latlon_to_utm = Transformer.from_crs('EPSG:4326', sim.crs_utm, always_xy=True)
start_lon, start_lat = float(waypoints_lonlat[0][0]), float(waypoints_lonlat[0][1])
sx, sy = latlon_to_utm.transform(start_lon, start_lat)

sim.waypoints = [waypoints_lonlat]
sim.spawn_speed = config['vessel']['initial_conditions']['speed_knots'] * 0.514444
sim.spawn()
# force pos
sim.pos[:, 0] = (sx, sy)

hull = sim._current_hull_poly(0)
print('Ship hull bounds:', hull.bounds)

# Test intersection between hull and each bridge/land geometry
from shapely.geometry import shape
found = False
for k in ('BRIDGE', 'LNDARE', 'COALNE'):
    layer = enc.get(k)
    if layer is None:
        continue
    if isinstance(layer, list):
        gdfs = layer
    else:
        gdfs = [layer]
    for gdf in gdfs:
        for geom in getattr(gdf, 'geometry', []):
            if geom is None:
                continue
            try:
                if hull.intersects(geom):
                    print('Intersection with', k)
                    print('  geom bounds:', getattr(geom, 'bounds', None))
                    print('  inter area/len:', getattr(hull.intersection(geom), 'area', None), getattr(hull.intersection(geom), 'length', None))
                    found = True
            except Exception as e:
                print('Geometry op error for', k, e)

if not found:
    print('No intersection found between hull and ENC geometries — likely CRS mismatch or geometries are far apart.')
