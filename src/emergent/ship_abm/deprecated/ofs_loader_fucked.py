# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Mon Jul 14 20:00:09 2025

@author: Kevin.Nebiolo
"""

"""
emergent.ship_abm.ofs_loader
--------------------------------
Utility that delivers a fast (u,v) sampler for any port in
`config.SIMULATION_BOUNDS`, using the most‑recent 2‑D surface‑current
file from the corresponding NOAA Operational Forecast System (OFS).
Falls back to global RTOFS if the regional grid isn’t available.

A single public helper is exposed:

    >>> sample = get_current_fn("Galveston")
    >>> uv = sample(lon, lat, dt.datetime.utcnow())   # (N,2) m/s

The module holds no external state; you can unit‑test `get_current_fn()`
independently of the rest of the ABM.
"""

import datetime as dt
from datetime import date, timedelta
from typing import Callable, Iterable, Tuple

import fsspec
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pyproj import Transformer

from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
BUCKETS: tuple[str, ...] = ("noaa-nos-ofs-pds", "noaa-ofs-pds")
CYCLES:  tuple[int, ...] = (18, 15, 12, 9, 6, 3, 0, 21)   # latest first
LAYERS:  tuple[str, ...] = ("n000")#, "f000")               # nowcast | 0‑h fcst

# Anonymous read‑only S3 filesystem
fs = fsspec.filesystem("s3", anon=True)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def regional_keys(model: str, day: date) -> Iterable[str]:
    """Yield every plausible *key* (sans bucket prefix) for *model* on *day*."""
    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
    ymd = f"{y}{m}{d}"
    for cyc in CYCLES:
        for layer in LAYERS:
            name = f"{model}.t{cyc:02d}z.{ymd}.2ds.{layer}.nc"
            # canonical layout (YYYY/MM/DD)
            yield f"{model}/netcdf/{y}/{m}/{d}/{name}"
            # legacy layout (YYYYMM/) kept during 2024 transition
            yield f"{model}/netcdf/{y}{m}/{name}"


def candidate_urls(model: str, day: date) -> list[str]:
    keys = list(regional_keys(model.lower(), day))
    return [f"s3://{bucket}/{key}" for bucket in BUCKETS for key in keys]


def first_existing_url(urls: list[str]) -> str | None:
    for url in urls:
        bucket, key = url[5:].split("/", 1)  # strip "s3://"
        if fs.exists(f"{bucket}/{key}"):
            return url
    return None

# ----------------------------------------------------------------------
# Open + subset helper
# ----------------------------------------------------------------------

def open_ofs_subset(
    model: str,
    start: date,
    bbox: Tuple[float, float, float, float],  # lon_min, lon_max, lat_min, lat_max
) -> xr.Dataset:
    """Open latest surface‑current file ≤14 days old and crop to *bbox*."""

    print(f"[ofs_loader] ⇒ searching for {model.upper()} up to {start} (±14 d)")
    for offset, day in enumerate(start - timedelta(n) for n in range(0, 15)):
        print(f"[ofs_loader]   • day −{offset}: {day}")
        url = first_existing_url(candidate_urls(model, day))
        if url:
            print(f"[ofs_loader]   ↳ FOUND file for {day}")
            break
    else:
        raise FileNotFoundError(f"No {model.upper()} data found in last 14 days")

    print(f"[ofs_loader] → opening   {url}")
    ds = xr.open_dataset(
        fs.open(url),
        engine="h5netcdf",  # netCDF‑4 stored as HDF‑5
        chunks={"time": 1},
    )
    print(f"[ofs_loader]   dataset opened with vars: {list(ds.data_vars.keys())}")

    # ------------------------------------------------------------------
    # Pick (u, v) variable names – many variants across OFSes
    # ------------------------------------------------------------------
    var_pairs = [
        ("ua", "va"),
        ("us", "vs"),
        ("u",  "v"),
        ("water_u", "water_v"),
    ]
    east = [v for v, da in ds.data_vars.items() if "eastward" in da.attrs.get("standard_name", "")]
    north = [v for v, da in ds.data_vars.items() if "northward" in da.attrs.get("standard_name", "")]
    var_pairs.extend(zip(east, north))

    for var_u, var_v in var_pairs:
        if var_u in ds and var_v in ds:
            print(f"[ofs_loader]   🛈 using current vars ({var_u}, {var_v})")
            break
    else:
        raise KeyError(f"No current variables in {url}; found {list(ds.data_vars)}")

    ds = ds[[var_u, var_v]].rename({var_u: "u", var_v: "v"})

    # ------------------------------------------------------------------
    # Normalise coordinates → lon / lat (2‑D or 1‑D)
    # ------------------------------------------------------------------
    for lon_name, lat_name in (("lon", "lat"), ("lonc", "latc"), ("longitude", "latitude")):
        if lon_name in ds and lat_name in ds:
            ds = ds.rename({lon_name: "lon", lat_name: "lat"})
            print(f"[ofs_loader]   🛈 coordinates renamed {lon_name}/{lat_name} → lon/lat")
            break

    if {"lon", "lat"} <= set(ds):
        ds = ds.set_coords(["lon", "lat"])

    print(f"[ofs_loader] ✔ opened    {url}")
    print(f"[ofs_loader]   grid dims: {ds.dims}")

    # ------------------------------------------------------------------
    # Spatial crop – works for both structured & unstructured
    # ------------------------------------------------------------------
    lon_min, lon_max, lat_min, lat_max = bbox

    if "lon" in ds and ds.lon.ndim == 2:
        before = ds.dims.get("x", ds.dims.get("xi_rho", 0)) * ds.dims.get("y", ds.dims.get("eta_rho", 0))
        ds = ds.where(
            (ds.lon > lon_min) & (ds.lon < lon_max) &
            (ds.lat > lat_min) & (ds.lat < lat_max),
            drop=True,
        )
        after = ds["u"].sizes.get("x", ds["u"].sizes.get("xi_rho", 0)) * ds["u"].sizes.get("y", ds["u"].sizes.get("eta_rho", 0))
        print(f"[ofs_loader]   🛈 structured crop: {before} → {after} cells")
    elif {"lon", "lat"} <= set(ds.coords):
        print(f"[ofs_loader]   🛈 unstructured grid with {ds.dims.get('node', 'unknown')} nodes – no bbox crop")

    return ds


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_current_fn(
    port: str,
    start: dt.datetime | None = None,
) -> Callable[[np.ndarray, np.ndarray, dt.datetime], np.ndarray]:
    """Return a callable that samples surface currents in m/s."""
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not found in SIMULATION_BOUNDS")

    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
    lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))

    # Inputs may be in UTM – detect & convert once
    if abs(lon_max) > 180 or abs(lat_max) > 90:
        utm_zone = int(((lon_min + lon_max) / 2 + 180) // 6) + 1
        utm_epsg = 32600 + utm_zone
        to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
        lon_min, lat_min = to_ll.transform(cfg["minx"], cfg["miny"])
        lon_max, lat_max = to_ll.transform(cfg["maxx"], cfg["maxy"])
        lon_min, lon_max = sorted((lon_min, lon_max))
        lat_min, lat_max = sorted((lat_min, lat_max))

    bbox = (lon_min, lon_max, lat_min, lat_max)
    print(f"[ofs_loader] bbox for {port}: "
          f"({lon_min:.4f}, {lat_min:.4f})–({lon_max:.4f}, {lat_max:.4f})")

    if start is None:
        start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    model = OFS_MODEL_MAP.get(port, "rtofs").lower()

    try:
        ds = open_ofs_subset(model, start.date(), bbox)
    except FileNotFoundError:
        ds = open_ofs_subset("rtofs", start.date(), bbox)

    times = ds.time.values
    is_fvcom = {"node"} <= set(ds.dims)

    if is_fvcom:
        # build KD‑tree once for (lon,lat) nodes
        xy = np.column_stack((ds.lon.values, ds.lat.values))
        tree = cKDTree(xy)
        u = ds["u"].isel(siglay=-1)  # surface layer
        v = ds["v"].isel(siglay=-1)

        def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
            idx_time = np.abs(times - np.datetime64(when)).argmin()
            idx_node = tree.query(np.column_stack((lon, lat)), k=1)[1]
            return np.column_stack((u[idx_time, idx_node].values,
                                    v[idx_time, idx_node].values))
    else:
        def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
            arr = ds.interp(
                time=np.datetime64(when),
                lon=("obs", lon),
                lat=("obs", lat),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
            return np.column_stack((arr.u.values, arr.v.values))

    return sample