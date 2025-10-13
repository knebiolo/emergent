"""Headless probe to sample the current_fn for a port and print diagnostics.

Usage:
    python scripts\probe_current.py --port Baltimore --n 10

This script will call `get_current_fn(port)` from `src/emergent/ship_abm/ofs_loader.py`
(via the package) and sample a small grid of points inside the port bounding box.
"""

import argparse
from datetime import datetime, timezone
import numpy as np

from emergent.ship_abm import ofs_loader
from emergent.ship_abm.config import SIMULATION_BOUNDS


def sample_port(port: str, n: int = 10):
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not found in SIMULATION_BOUNDS")
    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg['minx'], cfg['maxx']))
    lat_min, lat_max = sorted((cfg['miny'], cfg['maxy']))

    # create n random sample points inside bbox (deterministic seed)
    rng = np.random.default_rng(seed=12345)
    lons = rng.uniform(lon_min, lon_max, size=n)
    lats = rng.uniform(lat_min, lat_max, size=n)

    print(f"[probe_current] Sampling {n} points in {port} bbox: ({lon_min},{lat_min})–({lon_max},{lat_max})")
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # obtain sampler (may hit S3)
    try:
        sampler = ofs_loader.get_current_fn(port, start=now)
    except Exception as e:
        print(f"[probe_current] Failed to get_current_fn: {e}")
        raise

    # Annotate sampler if it has metadata
    meta = {}
    for attr in ('_native', '_u_pts', '_v_pts', '_valid_pts'):
        if hasattr(sampler, attr):
            meta[attr] = getattr(sampler, attr)
    if meta:
        print(f"[probe_current] sampler metadata keys: {list(meta.keys())}")

    # sample
    try:
        out = sampler(lons, lats, now)
    except Exception as e:
        print(f"[probe_current] sampler call failed: {e}")
        raise

    arr = np.asarray(out)
    print(f"[probe_current] sampler returned shape={arr.shape}")
    # normalize to (N,2)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape((1,2))
    if arr.ndim == 2 and arr.shape[1] >= 2:
        u = arr[:,0]
        v = arr[:,1]
    elif arr.ndim == 2 and arr.shape[0] == 2:
        u = arr[0,:]
        v = arr[1,:]
    else:
        flat = arr.ravel()
        if flat.size >= 2:
            u = np.full(n, float(flat[0]))
            v = np.full(n, float(flat[1]))
        else:
            u = np.zeros(n)
            v = np.zeros(n)

    def heading_deg(u_, v_):
        # Heading of flow in degrees (meteorological convention: coming from?)
        # We compute vector direction as atan2(v,u) → degrees from east CCW
        ang = (np.degrees(np.arctan2(v_, u_)) + 360.0) % 360.0
        return ang

    angs = heading_deg(u, v)

    print("[probe_current] Stats:")
    print(f"  u: min={np.nanmin(u):.3f} max={np.nanmax(u):.3f} mean={np.nanmean(u):.3f}")
    print(f"  v: min={np.nanmin(v):.3f} max={np.nanmax(v):.3f} mean={np.nanmean(v):.3f}")
    print(f"  speed: min={np.nanmin(np.hypot(u,v)):.3f} max={np.nanmax(np.hypot(u,v)):.3f} mean={np.nanmean(np.hypot(u,v)):.3f}")

    print("[probe_current] First samples (lon,lat) -> (u,v) m/s -> heading°:")
    for i in range(min(n, 20)):
        print(f"  {lons[i]:.5f},{lats[i]:.5f} -> {u[i]:6.3f},{v[i]:6.3f} -> {angs[i]:6.1f}°")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', default='Baltimore')
    parser.add_argument('--n', '-n', type=int, default=10)
    args = parser.parse_args()
    sample_port(args.port, args.n)
