"""Diagnose currents across a port by sampling a regular grid and computing divergence.

Usage:
  python scripts\diagnose_currents.py --port Baltimore --nx 40 --ny 40

Prints summary statistics and top positive/negative divergence hotspots (m/s per m).
"""
import argparse
from datetime import datetime, timezone
import numpy as np
from pyproj import Transformer

from emergent.ship_abm import ofs_loader
from emergent.ship_abm.config import SIMULATION_BOUNDS


def main(port: str, nx: int, ny: int):
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not in SIMULATION_BOUNDS")
    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg['minx'], cfg['maxx']))
    lat_min, lat_max = sorted((cfg['miny'], cfg['maxy']))

    # build regular lon/lat grid
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack((LON.ravel(), LAT.ravel()))

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    print(f"[diagnose] Sampling {nx}x{ny} grid for {port} at {now.isoformat()}")

    sampler = ofs_loader.get_current_fn(port, start=now)
    arr = sampler(pts[:,0], pts[:,1], now)
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[0] == pts.shape[0] and arr.shape[1] >= 2:
        U = arr[:,0].reshape((ny, nx))
        V = arr[:,1].reshape((ny, nx))
    elif arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == pts.shape[0]:
        U = arr[0,:].reshape((ny, nx))
        V = arr[1,:].reshape((ny, nx))
    else:
        raise RuntimeError(f"Unexpected sampler shape {arr.shape}")

    # Transform lon/lat grid to UTM meters for spacing
    lonc = 0.5*(lon_min + lon_max)
    utm_zone = int((lonc + 180) // 6) + 1
    utm_epsg = 32600 + utm_zone
    print(f"[diagnose] Using UTM zone {utm_zone} (EPSG:{utm_epsg}) for metric spacing")
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}', always_xy=True)
    X, Y = tr.transform(LON, LAT)

    # compute derivatives dudx and dvdy using central differences where possible
    # grid spacing varies, so compute gradients with respect to X and Y
    dudx = np.gradient(U, axis=1)
    # gradient returns dU/dx in units per index; convert by dividing by physical spacing
    # compute spacing arrays
    dx = np.gradient(X, axis=1)
    dy = np.gradient(Y, axis=0)
    # avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        dudx = dudx / dx
        dvdy = np.gradient(V, axis=0) / dy
        div = dudx + dvdy

    # flatten and summarize
    div_flat = div.ravel()
    U_flat = U.ravel(); V_flat = V.ravel(); speed = np.hypot(U_flat, V_flat)

    print(f"[diagnose] u: min={np.nanmin(U_flat):.4f} max={np.nanmax(U_flat):.4f} mean={np.nanmean(U_flat):.4f}")
    print(f"[diagnose] v: min={np.nanmin(V_flat):.4f} max={np.nanmax(V_flat):.4f} mean={np.nanmean(V_flat):.4f}")
    print(f"[diagnose] speed: min={np.nanmin(speed):.4f} max={np.nanmax(speed):.4f} mean={np.nanmean(speed):.4f}")
    print(f"[diagnose] divergence: min={np.nanmin(div_flat):.6e} max={np.nanmax(div_flat):.6e} mean={np.nanmean(div_flat):.6e}")

    # show top hotspots
    K = 10
    idx_pos = np.argsort(-div_flat)[:K]
    idx_neg = np.argsort(div_flat)[:K]
    print('\nTop positive divergence (sources/outflow):')
    for i in idx_pos:
        lon, lat = pts[i]
        print(f"  {lon:.5f},{lat:.5f}: div={div_flat[i]:.3e}, u={U_flat[i]:.3f}, v={V_flat[i]:.3f}")
    print('\nTop negative divergence (sinks/inflow):')
    for i in idx_neg:
        lon, lat = pts[i]
        print(f"  {lon:.5f},{lat:.5f}: div={div_flat[i]:.3e}, u={U_flat[i]:.3f}, v={V_flat[i]:.3f}")

    # optionally save files

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', '-p', default='Baltimore')
    p.add_argument('--nx', type=int, default=40)
    p.add_argument('--ny', type=int, default=40)
    args = p.parse_args()
    main(args.port, args.nx, args.ny)
