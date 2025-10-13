"""Sample the current sampler at a single lon/lat and print u,v and heading.

Usage:
  python scripts\sample_current_point.py --port Baltimore --lon -76.5 --lat 39.3
"""
import argparse
from datetime import datetime, timezone
import numpy as np
from emergent.ship_abm import ofs_loader


def heading_deg(u, v):
    return (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', '-p', default='Baltimore')
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--lat', type=float, required=True)
    args = p.parse_args()

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    try:
        sampler = ofs_loader.get_current_fn(args.port, start=now)
    except Exception as e:
        print(f"Failed to get_current_fn: {e}")
        raise
    try:
        out = sampler(np.array([args.lon]), np.array([args.lat]), now)
    except Exception as e:
        print(f"Sampler call failed: {e}")
        raise
    arr = np.asarray(out)
    if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
        u, v = float(arr[0,0]), float(arr[0,1])
    else:
        flat = arr.ravel()
        if flat.size >= 2:
            u, v = float(flat[0]), float(flat[1])
        else:
            u, v = 0.0, 0.0
    speed = np.hypot(u, v)
    ang = heading_deg(u, v)
    print(f"Sample at {args.lon:.6f},{args.lat:.6f} → u={u:.4f} m/s, v={v:.4f} m/s, speed={speed:.4f} m/s, heading={ang:.1f}°")
