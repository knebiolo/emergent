import datetime as dt
import numpy as np

try:
    from emergent.ship_abm.atmospheric import wind_sampler
    from emergent.ship_abm.config import SIMULATION_BOUNDS
except Exception as e:
    print('Import error:', e)
    raise

port = 'Baltimore' if 'Baltimore' in SIMULATION_BOUNDS else list(SIMULATION_BOUNDS.keys())[0]
cfg = SIMULATION_BOUNDS[port]
bbox = (cfg['minx'], cfg['maxx'], cfg['miny'], cfg['maxy'])
print('Using port', port, 'bbox', bbox)

ws = wind_sampler(bbox, dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0))
# sample 10x10 grid
lons = np.linspace(bbox[0], bbox[1], 10)
lats = np.linspace(bbox[2], bbox[3], 10)
pts = np.column_stack((np.repeat(lons, 10), np.tile(lats, 10)))
print('Querying', pts.shape[0], 'points')
vals = ws(pts[:, 0], pts[:, 1], dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0))
print('vals shape', vals.shape)
print('first 10 vals:\n', vals[:10])

# Also query a few individual points near first native station points if available
try:
    from emergent.ship_abm.atmospheric import build_wind_sampler
    print('build_wind_sampler available')
except Exception:
    pass

print('Done')
