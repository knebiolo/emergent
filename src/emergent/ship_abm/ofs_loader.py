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
from typing import Callable, Iterable, Tuple, List

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
LAYERS:  tuple[str, ...] = ("n000",)#, "f000")               # nowcast | 0‑h fcst

# Anonymous read‑only S3 filesystem
fs = fsspec.filesystem("s3", anon=True, requester_pays=True)


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
            # # legacy layout (YYYYMM/) kept during 2024 transition
            # yield f"{model}/netcdf/{y}{m}/{name}"

def candidate_urls(model: str, day: date) -> List[str]:
    # materialize the key-generator so we can len() it and reuse it
    keys_list = list(regional_keys(model, day))
    print(f"[ofs_loader]   got {len(keys_list)} keys for model={model} day={day}")  # DEBUG
    print(f"[ofs_loader]   BUCKETS = {BUCKETS}")  # DEBUG
    urls = [
        f"s3://{bucket}/{key}"
        for key in keys_list
        for bucket in BUCKETS
    ]
    print(f"[ofs_loader]   generated {len(urls)} candidate URLs")  # DEBUG
    return urls

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

    # search back 14 days for the freshest file that exists
    for day in (start - timedelta(n) for n in range(0, 15)):
        url = first_existing_url(candidate_urls(model, day))
        if url:
            break
    else:
        raise FileNotFoundError(f"No {model.upper()} data found in last 14 days")

    print(f"[ofs_loader] → opening   {url}")
    ds = xr.open_dataset(
        fs.open(url),
        engine="h5netcdf",  # netCDF‑4 stored as HDF‑5
        chunks={"time": 1},
    )

    # ------------------------------------------------------------------
    # Pick (u, v) variable names – many variants across OFSes
    # ------------------------------------------------------------------
    var_pairs = [
        ("ua", "va"),      # FVCOM / ROMS depth‑averaged
        ("us", "vs"),      # surface currents
        ("u",  "v"),
        ("water_u", "water_v"),  # RTOFS style
    ]
    # add any CF‑compliant names
    east = [v for v, da in ds.data_vars.items() if "eastward" in da.attrs.get("standard_name", "")]
    north = [v for v, da in ds.data_vars.items() if "northward" in da.attrs.get("standard_name", "")]
    var_pairs.extend(zip(east, north))

    for var_u, var_v in var_pairs:
        if var_u in ds and var_v in ds:
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
            break

    # promote to coords so they’re query‑able
    if {"lon", "lat"} <= set(ds):
        ds = ds.set_coords(["lon", "lat"])

    print(f"[ofs_loader] ✔ opened    {url}")

    # ------------------------------------------------------------------
    # Spatial crop – works for both structured & unstructured
    # ------------------------------------------------------------------
    lon_min, lon_max, lat_min, lat_max = bbox

    if "lon" in ds and ds.lon.ndim == 2:
        # structured 2‑D grid
        ds = ds.where(
            (ds.lon > lon_min) & (ds.lon < lon_max) &
            (ds.lat > lat_min) & (ds.lat < lat_max),
            drop=True,
        )
    elif {"lon", "lat"} <= set(ds.coords):
        # unstructured – 1‑D lon/lat; keep all, we’ll sample by KD‑tree later
        pass

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
        
    # extract time axis
    times = ds.time.values

    # Unstructured 1-D grid (nele or FVCOM node) → KD-tree sampling
    if ds.lon.ndim == 1:
        xy = np.column_stack((ds.lon.values, ds.lat.values))
        tree = cKDTree(xy)
        # if FVCOM, pick the top (surface) layer; otherwise use u/v as is
        if "siglay" in ds["u"].dims:
            u = ds["u"].isel(siglay=-1)
            v = ds["v"].isel(siglay=-1)
        else:
            u = ds["u"]
            v = ds["v"]
        
        # ─────────── NEW: materialize into NumPy once ───────────
        # bring the whole (ntime × npoints) arrays into memory
        u_arr = u.values   # shape (ntime, npts)
        v_arr = v.values
        # free the xarray objects so we don’t accidentally use them
        del u, v
    
        def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
            # find the nearest time‐slice and node‐indices
            idx_time = np.abs(times - np.datetime64(when)).argmin()
            idx_node = tree.query(np.column_stack((lon, lat)), k=1)[1]
            # index the pre‐loaded NumPy arrays directly
            return np.column_stack((
                u_arr[idx_time, idx_node],
                v_arr[idx_time, idx_node],
            ))
        
    else:
        # Structured 2-D grid → xarray interpolation
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


def candidate_met_urls(model: str, day: date) -> list[str]:
    """
    Yield every plausible S3 URL for *model*’s surface‐wind forcing  
    (both “.met.nowcast” and “.stations.nowcast” variants) on *day*.
    """
    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
    ymd = f"{y}{m}{d}"
    suffixes = [".met.nowcast.nc", ".stations.nowcast.nc"]
    urls: list[str] = []
    for cyc in CYCLES:
        for suf in suffixes:
            fname = f"{model}.t{cyc:02d}z.{ymd}{suf}"
            key   = f"{model}/netcdf/{y}/{m}/{d}/{fname}"
            for bucket in BUCKETS:
                urls.append(f"s3://{bucket}/{key}")
    print(f"[ofs_loader]   wind candidates: {len(urls)} URLs for model={model} day={day}")
    return urls


def open_met_subset(
    model: str,
    start: date,
    bbox: Tuple[float, float, float, float],
) -> xr.Dataset:
    """Open latest .met.nowcast file ≤14 days old and crop to *bbox*."""
    for day in (start - timedelta(n) for n in range(0, 15)):
        url = first_existing_url(candidate_met_urls(model, day))
        if url:
            break
    else:
        raise FileNotFoundError(f"No {model.upper()} MET data found in last 14 days")

    print(f"[ofs_loader] → opening wind {url}")
    # drop the problematic 'siglay' variable at load time
    ds_w = xr.open_dataset(
        fs.open(url),
        engine="h5netcdf",
        chunks={"time": 1},
        drop_variables=["siglay", "siglev"],  # drop vars whose names collide with dims
    )
    # rename lonc/latc → lon/lat for consistency
    if "lonc" in ds_w and "latc" in ds_w:
        ds_w = ds_w.rename({"lonc": "lon", "latc": "lat"})
    ds_w = ds_w.set_coords(["lon", "lat"])
    # drop everything except horizontal wind vars (you’ll need to adjust names)
    ds_w = ds_w[[v for v in ds_w.data_vars if "wind" in v.lower()]]

    # spatial crop: for unstructured 1-D, we just leave it for KD-tree
    if "lon" in ds_w and ds_w.lon.ndim == 2:
        lon_min, lon_max, lat_min, lat_max = bbox
        ds_w = ds_w.where(
            (ds_w.lon > lon_min) & (ds_w.lon < lon_max) &
            (ds_w.lat > lat_min) & (ds_w.lat < lat_max),
            drop=True,
        )
    return ds_w

def get_wind_fn(
    port: str,
    start: dt.datetime | None = None,
) -> Callable[[np.ndarray, np.ndarray, dt.datetime], np.ndarray]:
    """Return a (u,v) sampler for the model’s surface wind forcing."""
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not found")
    if start is None:
        start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    model = OFS_MODEL_MAP.get(port, "rtofs").lower()
    # reuse the same bbox logic as get_current_fn
    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
    lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))
    bbox = (lon_min, lon_max, lat_min, lat_max)

    ds_w = open_met_subset(model, start.date(), bbox)

    # pick variables: assume named 'uwind'/'vwind' or similar
    var_u, var_v = next(
        (u, v)
        for u, v in [("uwind_speed", "vwind_speed"), ("wind_u", "wind_v"), ("u", "v")]
        if u in ds_w and v in ds_w
    )
    ds_w = ds_w[[var_u, var_v]].rename({var_u: "u", var_v: "v"})

    # build KD-tree + NumPy arrays
    times_w = ds_w.time.values
    xy_w = np.column_stack((ds_w.lon.values, ds_w.lat.values))
    tree_w = cKDTree(xy_w)
    u_arr_w = ds_w["u"].values
    v_arr_w = ds_w["v"].values

    def sample_wind(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
        idx_t = np.abs(times_w - np.datetime64(when)).argmin()
        idx_n = tree_w.query(np.column_stack((lons, lats)), k=1)[1]
        return np.column_stack((u_arr_w[idx_t, idx_n], v_arr_w[idx_t, idx_n]))

    return sample_wind
    # times = ds.time.values
    # is_fvcom = {"node"} <= set(ds.dims)

    # if is_fvcom:
    #     # build KD‑tree once for (lon,lat) nodes
    #     xy = np.column_stack((ds.lon.values, ds.lat.values))
    #     tree = cKDTree(xy)
    #     u = ds["u"].isel(siglay=-1)  # surface layer
    #     v = ds["v"].isel(siglay=-1)

    #     def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
    #         idx_time = np.abs(times - np.datetime64(when)).argmin()
    #         idx_node = tree.query(np.column_stack((lon, lat)), k=1)[1]
    #         return np.column_stack((u[idx_time, idx_node].values,
    #                                 v[idx_time, idx_node].values))
    # else:
    #     def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
    #         arr = ds.interp(
    #             time=np.datetime64(when),
    #             lon=("obs", lon),
    #             lat=("obs", lat),
    #             method="linear",
    #             kwargs={"fill_value": np.nan},
    #         )
    #         return np.column_stack((arr.u.values, arr.v.values))

    # return sample