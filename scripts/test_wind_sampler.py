import traceback
from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS
import datetime as dt

port = 'Galveston'
print('Bounds for', port, SIMULATION_BOUNDS[port])

try:
    wind_fn = get_wind_fn(port)
    current_fn = get_current_fn(port)
    cfg = SIMULATION_BOUNDS[port]
    lon = (cfg['minx'] + cfg['maxx']) / 2
    lat = (cfg['miny'] + cfg['maxy']) / 2
    now = dt.datetime.utcnow()
    print('Sampling lon,lat:', lon, lat, 'at', now)
    print('wind:', wind_fn([lon], [lat], now))
    print('current:', current_fn([lon], [lat], now))
except Exception as e:
    print('ERROR running wind/current sampler:')
    traceback.print_exc()
