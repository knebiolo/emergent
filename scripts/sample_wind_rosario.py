"""Sample the wind sampler for Rosario Strait and print u,v,magnitude (m/s).
"""
from emergent.ship_abm.ofs_loader import get_wind_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS
import datetime as dt
import numpy as np

port = 'Rosario Strait'
cfg = SIMULATION_BOUNDS[port]
lon = (cfg['minx'] + cfg['maxx']) / 2.0
lat = (cfg['miny'] + cfg['maxy']) / 2.0
sampler = get_wind_fn(port)
when = dt.datetime.utcnow()
uv = sampler(np.array([lon]), np.array([lat]), when)
u, v = float(uv[0,0]), float(uv[0,1])
speed = np.hypot(u, v)
print(f"Wind sample @ {lon:.5f},{lat:.5f} UTC {when.isoformat()} -> u={u:.3f} m/s v={v:.3f} m/s speed={speed:.3f} m/s")

# sample a small grid of points across the channel
pts = []
for dx in (-0.05, 0.0, 0.05):
    p_lon = lon + dx
    uv = sampler(np.array([p_lon]), np.array([lat]), when)
    u, v = float(uv[0,0]), float(uv[0,1])
    pts.append((p_lon, u, v, np.hypot(u, v)))
print('Nearby samples (lon, u, v, speed):')
for p in pts:
    print(f'  {p[0]:.5f}  {p[1]:.3f} {p[2]:.3f}  {p[3]:.3f} m/s')
