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
indepe    print(f"[ofs_loader] → opening wind {url}")
    # drop the problematic 'siglay' variable at load time
    ds_w = xr.open_dataset(
        fs.open(url),
        engine="h5netcdf",
        chunks={"time": 1},
        drop_variables=["siglay", "siglev"],  # drop vars whose names collide with dims
    )
    ds_w.attrs["source_desc"] = f"NOAA OFS MET {model}"
    ds_w.attrs["source_urls"] = [url]
    
    # Find coordinate names - could be lon/lat, lonc/latc, x/y, etc.
    lon_coord = None
    lat_coord = None
    for possible_lon in ["lon", "lonc", "longitude", "x"]:
        if possible_lon in ds_w.variables:
            lon_coord = possible_lon
            break
    for possible_lat in ["lat", "latc", "latitude", "y"]:
        if possible_lat in ds_w.variables:
            lat_coord = possible_lat
            break
    
    if lon_coord is None or lat_coord is None:
        raise ValueError(f"Cannot find lon/lat coordinates in {url}. Available: {list(ds_w.variables)}")
    
    # Rename to standard lon/lat if needed
    if lon_coord != "lon" or lat_coord != "lat":
        rename_dict = {}
        if lon_coord != "lon":
            rename_dict[lon_coord] = "lon"
        if lat_coord != "lat":
            rename_dict[lat_coord] = "lat"
        ds_w = ds_w.rename(rename_dict)
    
    ds_w = ds_w.set_coords(["lon", "lat"])
    
    # drop everything except horizontal wind vars (you'll need to adjust names)
    ds_w = ds_w[[v for v in ds_w.data_vars if "wind" in v.lower()]] rest of the ABM.
"""

import datetime as dt
from datetime import date, timedelta
from typing import Callable, Iterable, Tuple, List

import fsspec
import os
import re
import tempfile
import shutil
import io
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pyproj import Transformer

from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
BUCKETS: tuple[str, ...] = ("noaa-nos-ofs-pds",)  # Only check primary bucket
CYCLES:  tuple[int, ...] = (18, 12, 6, 3, 0)       # Common OFS cycles (18z, 12z, 6z, 3z, 0z)
LAYERS:  tuple[str, ...] = ("n000",)#, "f000")     # nowcast | 0‑h fcst

# Anonymous read‑only S3 filesystem
fs = fsspec.filesystem("s3", anon=True, requester_pays=True)

# Cache monthly listings to avoid repeated S3 list calls
_MONTHLY_LIST_CACHE: dict[tuple[str, str], list[str]] = {}

_REGULARGRID_RE = re.compile(r"\.t(\d{2})z\.(\d{8})\.regulargrid\.f(\d{3})\.nc$")


def _list_monthly_paths(model: str, day: date) -> list[str]:
    """List S3 paths under the monthly archive for the given model/day."""
    y, m = day.strftime("%Y"), day.strftime("%m")
    prefix = f"{model}/netcdf/{y}{m}/"
    out: list[str] = []
    for bucket in BUCKETS:
        cache_key = (bucket, prefix)
        if cache_key not in _MONTHLY_LIST_CACHE:
            try:
                _MONTHLY_LIST_CACHE[cache_key] = fs.ls(f"{bucket}/{prefix}")
            except FileNotFoundError:
                _MONTHLY_LIST_CACHE[cache_key] = []
        out.extend(_MONTHLY_LIST_CACHE[cache_key])
    return out


def _monthly_urls_for_day(model: str, day: date) -> list[str]:
    """Return monthly-archive URLs for a specific day (current-related only)."""
    ymd = day.strftime("%Y%m%d")
    urls: list[str] = []
    for item in _list_monthly_paths(model, day):
        if not item or item.endswith("/"):
            continue
        if ymd not in item:
            continue
        name = os.path.basename(item)
        if not name.endswith(".nc"):
            continue
        if "regulargrid" in name or f"nos.{model}.fields" in name or name.startswith(f"{model}.t"):
            urls.append(f"s3://{item}" if not item.startswith("s3://") else item)
    return urls


def _iter_nearby_days(target: date, max_days: int) -> Iterable[date]:
    """Yield dates nearest to target (0, +1, -1, +2, -2, ...)."""
    yield target
    for offset in range(1, max_days + 1):
        yield target + timedelta(days=offset)
        yield target - timedelta(days=offset)


def _parse_regulargrid_time(url: str) -> dt.datetime | None:
    name = os.path.basename(url)
    match = _REGULARGRID_RE.search(name)
    if not match:
        return None
    cyc, ymd, fhr = match.groups()
    base = dt.datetime.strptime(f"{ymd}{cyc}", "%Y%m%d%H")
    return base + dt.timedelta(hours=int(fhr))


def _open_dataset_any(url: str, drop_vars: list[str] | None = None) -> xr.Dataset:
    """Open a remote OFS netCDF file, supporting both netCDF4 and netCDF3."""
    try:
        return xr.open_dataset(
            fs.open(url),
            engine="h5netcdf",
            chunks={"time": 1},
            drop_variables=drop_vars,
        )
    except (OSError, IOError, ValueError) as e_h5:
        # Detect netCDF3 via magic number and download locally for scipy
        with fs.open(url, "rb") as fin:
            magic = fin.read(4)

        tmp_path = None
        try:
            emsg = str(e_h5)
            print(f"[ofs_loader] [WARN] direct open failed: {emsg.encode('ascii','backslashreplace').decode('ascii')}; attempting tempfile fallback")
            suffix = os.path.splitext(url)[1] if os.path.splitext(url)[1] else ".nc"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
                tmp_path = tmpf.name
                with fs.open(url, "rb") as fin:
                    shutil.copyfileobj(fin, tmpf)

            if magic and magic.startswith(b"CDF"):
                # Windows keeps the tempfile locked while the dataset is open.
                # Load fully, close file handle, then return an in-memory dataset.
                with xr.open_dataset(tmp_path, engine="scipy", drop_variables=drop_vars) as ds_tmp:
                    return ds_tmp.load()

            with xr.open_dataset(
                tmp_path,
                engine="h5netcdf",
                chunks={"time": 1},
                drop_variables=drop_vars,
            ) as ds_tmp:
                return ds_tmp.load()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


def _select_uv_vars(ds: xr.Dataset, url: str) -> xr.Dataset:
    # Preserve horizontal coordinate variables that are often stored as data_vars
    coord_candidates = [
        "lon", "lat", "lonc", "latc", "longitude", "latitude", "Longitude", "Latitude",
        "lon_rho", "lat_rho", "lon_u", "lat_u", "lon_v", "lat_v",
    ]
    coord_to_set = [name for name in coord_candidates if name in ds and name not in ds.coords]
    if coord_to_set:
        ds = ds.set_coords(coord_to_set)

    # Variable aliases (depth-avg and surface)
    var_pairs = [
        ("u_sur", "v_sur"),     # WCOFS/ROMS surface-only files
        ("ua", "va"),
        ("us", "vs"),
        ("u", "v"),
        ("water_u", "water_v"),
        ("u_eastward", "v_northward"),
        ("u_2d", "v_2d"),
    ]
    # Add CF-compliant via standard_name if present
    east = [v for v, da in ds.data_vars.items() if "eastward" in da.attrs.get("standard_name", "")]
    north = [v for v, da in ds.data_vars.items() if "northward" in da.attrs.get("standard_name", "")]
    var_pairs.extend(zip(east, north))

    for var_u, var_v in var_pairs:
        if var_u in ds and var_v in ds:
            return ds[[var_u, var_v]].rename({var_u: "u", var_v: "v"})

    raise KeyError(f"No recognizable 2D current variables found in {url}. Available: {list(ds.data_vars)}")


def _normalize_time_coord(ds: xr.Dataset, file_time: dt.datetime | None = None) -> xr.Dataset:
    # Rename time dimension if needed
    if "ocean_time" in ds.dims:
        ds = ds.rename({"ocean_time": "time"})
    elif "time" not in ds.dims:
        for d in ds.dims:
            if "time" in d.lower():
                ds = ds.rename({d: "time"})
                break

    if "time" in ds.coords:
        if file_time is not None and not np.issubdtype(ds["time"].dtype, np.datetime64):
            if ds["time"].size != 1:
                raise ValueError("Non-datetime time coordinate with size != 1 cannot be coerced.")
            ds = ds.assign_coords(time=[np.datetime64(file_time)])
    elif file_time is not None:
        ds = ds.expand_dims("time")
        ds = ds.assign_coords(time=[np.datetime64(file_time)])
    return ds


def _regularize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    """Convert separable 2D lon/lat grids into 1D coords for interp."""
    if "lon" not in ds or "lat" not in ds:
        return ds
    if ds.lon.ndim != 2 or ds.lat.ndim != 2:
        return ds
    if ds.lon.dims != ds.lat.dims or len(ds.lon.dims) != 2:
        return ds
    dim_y, dim_x = ds.lon.dims
    lon2 = ds.lon.values
    lat2 = ds.lat.values
    lon1 = lon2[0, :]
    lat1 = lat2[:, 0]
    diff_lon = np.nanmax(np.abs(lon2 - lon1[None, :]))
    diff_lat = np.nanmax(np.abs(lat2 - lat1[:, None]))
    if not (np.isfinite(diff_lon) and np.isfinite(diff_lat)):
        raise ValueError("Non-finite lon/lat values found while regularizing grid.")
    if diff_lon > 1e-6 or diff_lat > 1e-6:
        return ds
    ds = ds.drop_vars(["lon", "lat"])
    ds = ds.assign_coords(lon=(dim_x, lon1), lat=(dim_y, lat1))
    ds = ds.swap_dims({dim_x: "lon", dim_y: "lat"})
    if {"time", "lat", "lon"} <= set(ds.dims):
        ds = ds.transpose("time", "lat", "lon")
    return ds


def _crop_to_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    lon_min, lon_max, lat_min, lat_max = _convert_bbox_for_dataset(ds, bbox)
    if "lon" in ds and "lat" in ds:
        if ds.lon.ndim == 1 and ds.lat.ndim == 1:
            ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        elif ds.lon.ndim == 2:
            ds = ds.where(
                (ds.lon > lon_min) & (ds.lon < lon_max) &
                (ds.lat > lat_min) & (ds.lat < lat_max),
                drop=True,
            )
    return ds


def _normalize_current_ds(
    ds: xr.Dataset,
    bbox: Tuple[float, float, float, float],
    url: str,
    file_time: dt.datetime | None = None,
) -> xr.Dataset:
    ds = _select_uv_vars(ds, url)
    ds = _normalize_time_coord(ds, file_time=file_time)

    if "Depth" in ds.dims:
        ds = ds.isel(Depth=0)

    # Normalize coordinates to lon/lat if present
    for lon_name, lat_name in (("lon", "lat"), ("lonc", "latc"), ("longitude", "latitude"), ("Longitude", "Latitude")):
        if lon_name in ds and lat_name in ds:
            if lon_name != "lon" or lat_name != "lat":
                ds = ds.rename({lon_name: "lon", lat_name: "lat"})
            break

    if {"lon", "lat"} <= set(ds):
        ds = ds.set_coords(["lon", "lat"])
        ds = _regularize_lon_lat(ds)
        ds = _crop_to_bbox(ds, bbox)

    return ds


def _daily_average_current(ds: xr.Dataset, target_dt: dt.datetime) -> xr.Dataset:
    """Collapse time to a single daily-average snapshot and set time to target_dt."""
    if "time" in ds.dims:
        if ds.sizes.get("time", 1) > 1:
            ds = ds.mean(dim="time", skipna=True, keep_attrs=True)
        if "time" not in ds.dims:
            ds = ds.expand_dims("time")
    else:
        ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=[np.datetime64(target_dt)])
    ds.attrs["daily_average"] = True
    return ds


def _open_ofs_time_series(
    model: str,
    start_dt: dt.datetime,
    bbox: Tuple[float, float, float, float],
    time_window_hours: float,
) -> xr.Dataset | None:
    """Attempt to open a time-varying regulargrid series from monthly archives."""
    if time_window_hours <= 0:
        return None
    pad = dt.timedelta(hours=1)
    window_start = start_dt - pad
    window_end = start_dt + dt.timedelta(hours=time_window_hours)

    start_day = window_start.date()
    end_day = window_end.date()
    urls: list[tuple[dt.datetime, str]] = []
    for i in range((end_day - start_day).days + 1):
        day = start_day + dt.timedelta(days=i)
        for url in _monthly_urls_for_day(model, day):
            if "regulargrid" not in url:
                continue
            t = _parse_regulargrid_time(url)
            if t is None:
                continue
            if window_start <= t <= window_end:
                urls.append((t, url))

    if not urls:
        return None

    urls.sort(key=lambda x: x[0])
    ds_list: list[xr.Dataset] = []
    source_urls: list[str] = []
    for t, url in urls:
        ds = _open_dataset_any(url, drop_vars=["siglay", "siglev"])
        ds = _normalize_current_ds(ds, bbox, url=url, file_time=t)
        ds_list.append(ds)
        source_urls.append(url)

    if not ds_list:
        return None

    if len(ds_list) == 1:
        ds = ds_list[0]
    else:
        ds = xr.concat(ds_list, dim="time")
        if "time" in ds.coords:
            ds = ds.sortby("time")
            ds = ds.sel(time=slice(np.datetime64(window_start), np.datetime64(window_end)))

    ds.attrs["source_urls"] = source_urls
    ds.attrs["source_desc"] = f"NOAA OFS {model} regulargrid archive"
    ds.attrs["source_window"] = (window_start.isoformat(), window_end.isoformat())
    return ds


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def regional_keys(model: str, day: date) -> List[str]:
    """Yield plausible S3 *keys* (sans bucket) for *model* on *day*.
    Adds legacy/no-`netcdf` layouts to improve hit rate.
    """
    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
    ymd = f"{y}{m}{d}"
    keys: List[str] = []
    
    
    for cyc in CYCLES:
        if model.lower() == "rtofs":
            name = f"rtofs.t{cyc:02d}z.global.2ds.n000.nc"
            keys += [
            f"rtofs/{y}/{m}/{d}/{name}", # daily tree (current)
            f"rtofs/{y}{m}/{name}", # legacy monthly
            f"rtofs/netcdf/{y}/{m}/{d}/{name}", # historical netcdf tree
            ]
        else:
            # Different models use different naming:
            # - FVCOM models (cbofs, sscofs, etc): .fields.f000.nc
            # - ROMS models (ngofs2, wcofs, etc): .2ds.f000.nc
            # Try both patterns
            for suffix in ["fields.f000.nc", "fields.f001.nc", "2ds.f000.nc", "2ds.f001.nc"]:
                name = f"{model}.t{cyc:02d}z.{ymd}.{suffix}"
                keys.append(f"{model}/netcdf/{y}/{m}/{d}/{name}")
    return keys
        
def candidate_urls(model: str, day: date, include_monthly: bool = True) -> List[str]:
    keys_list = regional_keys(model, day)
    print(f"[ofs_loader] got {len(keys_list)} keys for model={model} day={day}")
    print(f"[ofs_loader] BUCKETS = {BUCKETS}")
    urls = [f"s3://{bucket}/{key}" for key in keys_list for bucket in BUCKETS]
    print(f"[ofs_loader] generated {len(urls)} candidate URLs")
    # Include monthly archive URLs (historical)
    if include_monthly:
        monthly_urls = _monthly_urls_for_day(model, day)
        if monthly_urls:
            urls.extend(monthly_urls)
    # de-duplicate while preserving order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    return urls  

def first_existing_url(urls: List[str]) -> str | None:
    """Check URLs in order, return first that exists."""
    for url in urls:
        if not url.startswith("s3://"):
            raise ValueError(f"Malformed URL (expected s3://): {url}")
        bucket_key = url[5:]
        if "/" not in bucket_key:
            raise ValueError(f"Malformed S3 URL (missing key): {url}")
        bucket, key = bucket_key.split("/", 1)
        if fs.exists(f"{bucket}/{key}"):
            print(f"[ofs_loader] [OK] Found: {url}")
            return url
    return None

# def candidate_urls(model: str, day: date) -> List[str]:
#     # materialize the key-generator so we can len() it and reuse it
#     keys_list = list(regional_keys(model, day))
#     print(f"[ofs_loader]   got {len(keys_list)} keys for model={model} day={day}")  # DEBUG
#     print(f"[ofs_loader]   BUCKETS = {BUCKETS}")  # DEBUG
#     urls = [
#         f"s3://{bucket}/{key}"
#         for key in keys_list
#         for bucket in BUCKETS
#     ]
#     print(f"[ofs_loader]   generated {len(urls)} candidate URLs")  # DEBUG
#     return urls

# def first_existing_url(urls: list[str]) -> str | None:
#     for url in urls:
#         bucket, key = url[5:].split("/", 1)  # strip "s3://"
#         if fs.exists(f"{bucket}/{key}"):
#             return url
#     return None

# ----------------------------------------------------------------------
# Open + subset helper
# ----------------------------------------------------------------------


def _convert_bbox_for_dataset(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert [-180,180] bbox to [0,360] if dataset longitudes use 0–360."""
    lon_min, lon_max, lat_min, lat_max = bbox
    # Heuristic: any lon > 180 implies 0–360 convention
    for lon_name in ("lon", "lonc", "longitude"):
        if lon_name in ds:
            if float(ds[lon_name].max()) > 180.0:
                to0360 = lambda L: L + 360.0 if L < 0.0 else L
                lon_min, lon_max = to0360(lon_min), to0360(lon_max)
            break
    return (lon_min, lon_max, lat_min, lat_max)

def open_ofs_subset(
model: str,
start,
bbox: Tuple[float, float, float, float], # lon_min, lon_max, lat_min, lat_max
time_window_hours: float = 0.0,
search_days: int = 14,
nearest: bool = False,
prefer_daily: bool = True,
) -> xr.Dataset:
    start_dt = start if isinstance(start, dt.datetime) else dt.datetime.combine(start, dt.time(0))

    # Try time-series first (regulargrid monthly archives)
    if time_window_hours and time_window_hours > 0:
        ds_ts = _open_ofs_time_series(model, start_dt, bbox, time_window_hours)
        if ds_ts is not None:
            return ds_ts

    # Find a single file near the requested day - STOP AT FIRST SUCCESS
    if nearest:
        day_iter = _iter_nearby_days(start_dt.date(), max_days=search_days)
    else:
        day_iter = (start_dt.date() - timedelta(n) for n in range(0, search_days + 1))

    for day in day_iter:
        if prefer_daily:
            # Prefer daily netCDF4 first; fall back to monthly archive only if needed
            url = first_existing_url(candidate_urls(model, day, include_monthly=False))
            if url:
                print('[ofs_loader] Opening: ' + str(url))
                ds = _open_dataset_any(url, drop_vars=['siglay', 'siglev'])
                ds = _normalize_current_ds(ds, bbox, url=url, file_time=None)
                ds.attrs['source_urls'] = [url]
                ds.attrs['source_desc'] = 'NOAA OFS ' + str(model)
                ds.attrs['source_date'] = day.isoformat()
                if nearest and day != start_dt.date():
                    ds.attrs['source_date_requested'] = start_dt.date().isoformat()
                print('[ofs_loader] [OK] Successfully opened dataset from ' + str(day))
                return ds

        # Only try monthly archives if no daily file was found (or prefer_daily=False)
        monthly_urls = _monthly_urls_for_day(model, day)
        if monthly_urls:
            def _pref_key(u: str):
                if f"nos.{model}.fields" in u:
                    return (0, u)
                if "fields" in u:
                    return (1, u)
                if "regulargrid" in u:
                    return (2, u)
                return (3, u)
            for url in sorted(monthly_urls, key=_pref_key):
                print('[ofs_loader] Opening: ' + str(url))
                ds = _open_dataset_any(url, drop_vars=['siglay', 'siglev'])
                ds = _normalize_current_ds(ds, bbox, url=url, file_time=None)
                ds.attrs['source_urls'] = [url]
                ds.attrs['source_desc'] = 'NOAA OFS ' + str(model)
                ds.attrs['source_date'] = day.isoformat()
                if nearest and day != start_dt.date():
                    ds.attrs['source_date_requested'] = start_dt.date().isoformat()
                print('[ofs_loader] [OK] Successfully opened dataset from ' + str(day))
                return ds

    scope = f"within {search_days} days of {start_dt.date().isoformat()}" if nearest else f"in last {search_days} days"
    raise FileNotFoundError('No ' + str(model).upper() + f' data could be opened {scope}')

# def open_ofs_subset(
#     model: str,
#     start: date,
#     bbox: Tuple[float, float, float, float],  # lon_min, lon_max, lat_min, lat_max
# ) -> xr.Dataset:
#     """Open latest surface‑current file ≤14 days old and crop to *bbox*."""

#     # search back 14 days for the freshest file that exists
#     for day in (start - timedelta(n) for n in range(0, 15)):
#         url = first_existing_url(candidate_urls(model, day))
#         if url:
#             break
#     else:
#         raise FileNotFoundError(f"No {model.upper()} data found in last 14 days")

#     print(f"[ofs_loader] → opening   {url}")
#     ds = xr.open_dataset(
#         fs.open(url),
#         engine="h5netcdf",  # netCDF‑4 stored as HDF‑5
#         chunks={"time": 1},
#     )

#     # ------------------------------------------------------------------
#     # Pick (u, v) variable names – many variants across OFSes
#     # ------------------------------------------------------------------
#     var_pairs = [
#         ("ua", "va"),      # FVCOM / ROMS depth‑averaged
#         ("us", "vs"),      # surface currents
#         ("u",  "v"),
#         ("water_u", "water_v"),  # RTOFS style
#     ]
#     # add any CF‑compliant names
#     east = [v for v, da in ds.data_vars.items() if "eastward" in da.attrs.get("standard_name", "")]
#     north = [v for v, da in ds.data_vars.items() if "northward" in da.attrs.get("standard_name", "")]
#     var_pairs.extend(zip(east, north))

#     for var_u, var_v in var_pairs:
#         if var_u in ds and var_v in ds:
#             break
#     else:
#         raise KeyError(f"No current variables in {url}; found {list(ds.data_vars)}")

#     ds = ds[[var_u, var_v]].rename({var_u: "u", var_v: "v"})

#     # ------------------------------------------------------------------
#     # Normalise coordinates → lon / lat (2‑D or 1‑D)
#     # ------------------------------------------------------------------
#     for lon_name, lat_name in (("lon", "lat"), ("lonc", "latc"), ("longitude", "latitude")):
#         if lon_name in ds and lat_name in ds:
#             ds = ds.rename({lon_name: "lon", lat_name: "lat"})
#             break

#     # promote to coords so they’re query‑able
#     if {"lon", "lat"} <= set(ds):
#         ds = ds.set_coords(["lon", "lat"])

#     print(f"[ofs_loader] ✔ opened    {url}")

#     # ------------------------------------------------------------------
#     # Spatial crop – works for both structured & unstructured
#     # ------------------------------------------------------------------
#     lon_min, lon_max, lat_min, lat_max = bbox

#     if "lon" in ds and ds.lon.ndim == 2:
#         # structured 2‑D grid
#         ds = ds.where(
#             (ds.lon > lon_min) & (ds.lon < lon_max) &
#             (ds.lat > lat_min) & (ds.lat < lat_max),
#             drop=True,
#         )
#     elif {"lon", "lat"} <= set(ds.coords):
#         # unstructured – 1‑D lon/lat; keep all, we’ll sample by KD‑tree later
#         pass

#     return ds


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
_M2_OMEGA = 2.0 * np.pi / (12.4206 * 3600.0) # rad/s
_S2_OMEGA = 2.0 * np.pi / (12.0000 * 3600.0) # rad/s


def make_tidal_proxy_current(axis_deg: float = 320.0,
A_M2: float = 0.30,
A_S2: float = 0.10,
phase_M2: float = 0.0,
phase_S2: float = 0.0) -> Callable:
    """Return f(lon, lat, when)->(N,2) m/s using a uniform M2+S2 proxy.
    axis_deg: principal axis in degrees (0=east, 90=north).
    Amplitudes in m/s. """
    theta = np.deg2rad(axis_deg)
    ex, ey = np.cos(theta), np.sin(theta)
    
    
    def _to_seconds(t):
        if hasattr(t, "timestamp"):
            return t.timestamp()
        return float(t)
    
    
    def current(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
        t = _to_seconds(when)
        speed = (A_M2 * np.sin(_M2_OMEGA * t + phase_M2) +
        A_S2 * np.sin(_S2_OMEGA * t + phase_S2))
        u = np.full_like(np.asarray(lons, dtype=float), speed * ex)
        v = np.full_like(np.asarray(lats, dtype=float), speed * ey)
        return np.column_stack((u, v))
        
    
    return current

def get_current_fn(
    port: str,
    start: dt.datetime | None = None,
    land_gdf=None,
    time_window_hours: float | None = None,
    daily_average: bool = False,
    search_days: int = 14,
    nearest: bool = False,
    prefer_daily: bool = True,
) -> Callable[[np.ndarray, np.ndarray, dt.datetime], np.ndarray]:
    """Return f(lon, lat, when)->(N,2) m/s.
    Tries regional OFS → RTOFS, then falls back to a tidal proxy.
    """
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not found in SIMULATION_BOUNDS")
        
    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
    lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))
    bbox = (lon_min, lon_max, lat_min, lat_max)
        
    if start is None:
        start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_dt = start
    if daily_average:
        window_hours = 0.0
    else:
        window_hours = time_window_hours if time_window_hours is not None else 0.0
        
    model = OFS_MODEL_MAP.get(port, "rtofs").lower()
    
    try:
        ds = open_ofs_subset(
            model,
            start_dt,
            bbox,
            time_window_hours=window_hours,
            search_days=search_days,
            nearest=nearest,
            prefer_daily=prefer_daily,
        )
        print(f"[ofs_loader] [OK] Using {model.upper()} data")
    except FileNotFoundError as e:
        print(f"[ofs_loader] WARN: {model.upper()} open failed ({str(e)}); trying RTOFS...")
        ds = open_ofs_subset(
            "rtofs",
            start_dt,
            bbox,
            time_window_hours=window_hours,
            search_days=search_days,
            nearest=nearest,
            prefer_daily=prefer_daily,
        )
        print(f"[ofs_loader] [OK] Using RTOFS data")

    if daily_average:
        ds = _daily_average_current(ds, start_dt)
        src = ds.attrs.get("source_desc")
        if src:
            ds.attrs["source_desc"] = f"{src} (daily avg)"
        requested = start_dt.date().isoformat()
        actual = ds.attrs.get("source_date")
        if actual and actual != requested:
            ds.attrs.setdefault("source_date_requested", requested)
            print(f"[ofs_loader] WARN: no data for {requested}; using nearest {actual}")
    
    def _attach_source(sampler, source_override: str | None = None):
        src = source_override
        if src is None:
            src = ds.attrs.get("source_desc")
        urls = ds.attrs.get("source_urls")
        date_used = ds.attrs.get("source_date")
        date_req = ds.attrs.get("source_date_requested")
        if src:
            sampler._source = src
        if urls:
            sampler._source_urls = urls
        if date_used:
            sampler._source_date = date_used
        if date_req:
            sampler._source_date_requested = date_req
        if window_hours and start_dt:
            end_dt = start_dt + dt.timedelta(hours=float(window_hours))
            sampler._source_window = (start_dt.isoformat(), end_dt.isoformat())
        return sampler

    # Build sampler from ds (structured vs unstructured)
    # If lon uses 0–360, we’ll convert query points on the fly.
    lon_0360 = False
    if "lon" in ds:
        lon_0360 = float(ds["lon"].max()) > 180.0
            
    if (
        "lon" in ds.dims and "lat" in ds.dims
        and ds.lon.ndim == 1 and ds.lat.ndim == 1
        and ds.lon.dims == ("lon",) and ds.lat.dims == ("lat",)
    ):
        # Structured 1-D rectilinear grid: bilinear interp
        def sample(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
            lons = np.asarray(lons, dtype=float)
            lats = np.asarray(lats, dtype=float)

            if lon_0360:
                lons = np.where(lons < 0.0, lons + 360.0, lons)
            arr = ds.interp(
                time=np.datetime64(when),
                lon=("points", lons),
                lat=("points", lats),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
            u = arr.u.values
            v = arr.v.values
            if u.ndim == 2:  # squeeze time
                u = u[0]
                v = v[0]
            return np.column_stack((u, v))
        # annotate sampler for downstream resampling/diagnostics
        sample._native = 'rectilinear'
        sample._lon_0360 = lon_0360
        return _attach_source(sample)

    if "lon" in ds and ds.lon.ndim == 2:
        # 2-D lon/lat coords on (ny,nx)-style grids are not valid interp dimensions
        # for xarray.interp. Use the KDTree branch below instead.
        print("[ofs_loader] Detected 2-D lon/lat grid - using nearest-neighbor sampler")
    
    # ROMS curvilinear C-grid: Use NearestNDInterpolator (FAST, no Delaunay)
    is_roms = "lon_rho" in ds or "lon_u" in ds
    if is_roms:
        from scipy.interpolate import NearestNDInterpolator
        
        print("[ofs_loader] Detected ROMS curvilinear grid - using nearest-neighbor interpolation")
        
        # Extract surface layer from 3D/4D data
        u_data = ds["u"]
        v_data = ds["v"]
        
        # Handle different vertical coordinate names
        vert_dims = [d for d in u_data.dims if d in ['siglay', 's_rho', 'z', 'depth', 'sigma']]
        time_dims = [d for d in u_data.dims if 'time' in d.lower()]
        
        # Select surface and first time
        if vert_dims:
            vert_dim = vert_dims[0]
            u_data = u_data.isel({vert_dim: -1})
            v_data = v_data.isel({vert_dim: -1})
        if time_dims:
            time_dim = time_dims[0]
            u_data = u_data.isel({time_dim: 0})
            v_data = v_data.isel({time_dim: 0})
        
        # U and V are on staggered C-grids - build separate trees for each
        lon_u = ds["lon_u"].values.ravel() if "lon_u" in ds else ds["lon_rho"].values.ravel()
        lat_u = ds["lat_u"].values.ravel() if "lat_u" in ds else ds["lat_rho"].values.ravel()
        lon_v = ds["lon_v"].values.ravel() if "lon_v" in ds else ds["lon_rho"].values.ravel()
        lat_v = ds["lat_v"].values.ravel() if "lat_v" in ds else ds["lat_rho"].values.ravel()
        
        # Get velocity arrays
        u_arr = u_data.values.ravel()
        v_arr = v_data.values.ravel()
        
        # Remove NaN values
        u_valid_mask = ~np.isnan(u_arr)
        v_valid_mask = ~np.isnan(v_arr)
        
        lon_u_valid = lon_u[u_valid_mask]
        lat_u_valid = lat_u[u_valid_mask]
        u_valid = u_arr[u_valid_mask]
        
        lon_v_valid = lon_v[v_valid_mask]
        lat_v_valid = lat_v[v_valid_mask]
        v_valid = v_arr[v_valid_mask]
        
        print(f"[ofs_loader] U-grid: {len(u_valid):,} valid points")
        print(f"[ofs_loader] V-grid: {len(v_valid):,} valid points")
        print(f"[ofs_loader] U: range=[{u_valid.min():.3f}, {u_valid.max():.3f}] m/s")
        print(f"[ofs_loader] V: range=[{v_valid.min():.3f}, {v_valid.max():.3f}] m/s")
        
        u_pts = np.column_stack((lon_u_valid, lat_u_valid))
        v_pts = np.column_stack((lon_v_valid, lat_v_valid))

        # Optionally filter out points that lie on land (ENC LNDARE) so
        # nearest-neighbour picks do not snap to inland grid points.
        if land_gdf is not None:
            from shapely.geometry import Point
            from shapely.ops import unary_union
            from shapely.prepared import prep

            land_union = unary_union(list(getattr(land_gdf, 'geometry', [])))
            land_prep = prep(land_union)

            u_pts_all = np.column_stack((lon_u_valid, lat_u_valid))
            v_pts_all = np.column_stack((lon_v_valid, lat_v_valid))

            u_on_land = np.array([land_prep.contains(Point(xy)) for xy in u_pts_all])
            v_on_land = np.array([land_prep.contains(Point(xy)) for xy in v_pts_all])

            u_keep = ~u_on_land
            v_keep = ~v_on_land

            # Apply masks, but fall back to original arrays if filtering removes all points
            if np.count_nonzero(u_keep) > 0:
                u_pts = u_pts_all[u_keep]
                u_valid = u_valid[u_keep]
            else:
                print("[ofs_loader] WARN: land filtering removed all U-grid points — keeping originals")
                u_pts = u_pts_all

            if np.count_nonzero(v_keep) > 0:
                v_pts = v_pts_all[v_keep]
                v_valid = v_valid[v_keep]
            else:
                print("[ofs_loader] WARN: land filtering removed all V-grid points — keeping originals")
                v_pts = v_pts_all

        # Build separate KDTrees for U and V grids (FAST!)
        u_tree = cKDTree(u_pts)
        v_tree = cKDTree(v_pts)
        
        print("[ofs_loader] Built FAST nearest-neighbor interpolators for staggered grids")
        
        def sample(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
            lons = np.asarray(lons, dtype=float)
            lats = np.asarray(lats, dtype=float)
            
            query_pts = np.column_stack((lons, lats))
            
            # Find nearest neighbors on each grid (FAST!)
            _, u_indices = u_tree.query(query_pts)
            _, v_indices = v_tree.query(query_pts)
            
            u_vals = u_valid[u_indices]
            v_vals = v_valid[v_indices]
            
            return np.column_stack((u_vals, v_vals))
        # annotate sampler with native grids and valid data for resampling
        sample._native = 'roms'
        sample._u_pts = u_pts
        sample._v_pts = v_pts
        sample._u_valid = u_valid
        sample._v_valid = v_valid
        return _attach_source(sample)
    
    # Unstructured FVCOM grid: use fast nearest-neighbor (NO slow Delaunay!)
    from scipy.interpolate import NearestNDInterpolator
    
    print("[ofs_loader] Detected unstructured FVCOM grid - using fast nearest-neighbor")
    
    # FVCOM: u/v are on element centers (lonc, latc), not nodes (lon, lat)
    if "lonc" in ds and "latc" in ds:
        lon_coords = ds["lonc"].values.ravel()
        lat_coords = ds["latc"].values.ravel()
        print("[ofs_loader] Using element-center coordinates (lonc, latc)")
    elif "lon" in ds and "lat" in ds:
        lon_coords = ds["lon"].values.ravel()
        lat_coords = ds["lat"].values.ravel()
        print("[ofs_loader] Using node coordinates (lon, lat)")
    else:
        raise KeyError("No lon/lat coordinates found in dataset")
    
    if lon_0360:
        lon_coords = np.where(lon_coords < 0.0, lon_coords + 360.0, lon_coords)
        
    pts = np.column_stack((lon_coords, lat_coords))

    # Optionally filter out points that lie on land (if ENC LNDARE provided)
    if land_gdf is not None:
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep

        land_union = unary_union(list(getattr(land_gdf, 'geometry', [])))
        land_prep = prep(land_union)

        on_land = np.array([land_prep.contains(Point(xy)) for xy in pts])
        keep = ~on_land
        if np.count_nonzero(keep) > 0:
            valid_pts = pts[keep]
        else:
            print("[ofs_loader] WARN: land filtering removed all FVCOM points — keeping originals")
            valid_pts = pts
    else:
        valid_pts = pts

    tree = cKDTree(valid_pts)
    
    # Extract surface layer from 3D/4D data
    u_data = ds["u"]
    v_data = ds["v"]
    
    # Handle different vertical coordinate names
    vert_dims = [d for d in u_data.dims if d in ['siglay', 's_rho', 'z', 'depth', 'sigma']]
    time_dims = [d for d in u_data.dims if 'time' in d.lower()]
    
    # Select surface and first time
    if vert_dims:
        vert_dim = vert_dims[0]
        u_data = u_data.isel({vert_dim: -1})
        v_data = v_data.isel({vert_dim: -1})
    if time_dims:
        time_dim = time_dims[0]
        u_data = u_data.isel({time_dim: 0})
        v_data = v_data.isel({time_dim: 0})
    
    u_flat = u_data.values.ravel()
    v_flat = v_data.values.ravel()
    
    # Remove NaN values for KDTree
    valid = ~(np.isnan(u_flat) | np.isnan(v_flat))
    lon_valid = lon_coords[valid]
    lat_valid = lat_coords[valid]
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    
    print(f"[ofs_loader] Surface data: {len(u_valid):,} valid points (removed {np.sum(~valid):,} NaN)")
    print(f"[ofs_loader] U: range=[{np.nanmin(u_valid):.3f}, {np.nanmax(u_valid):.3f}] m/s")
    print(f"[ofs_loader] V: range=[{np.nanmin(v_valid):.3f}, {np.nanmax(v_valid):.3f}] m/s")
    
    # Build fast KDTree (no Delaunay triangulation!)
    valid_pts = np.column_stack((lon_valid, lat_valid))

    # If land_gdf was provided earlier for the FVCOM/unstructured branch, try to filter
    if land_gdf is not None:
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep

        land_union = unary_union(list(getattr(land_gdf, 'geometry', [])))
        land_prep = prep(land_union)
        on_land = np.array([land_prep.contains(Point(xy)) for xy in valid_pts])
        keep = ~on_land
        if np.count_nonzero(keep) > 0:
            valid_pts = valid_pts[keep]
            u_valid = u_valid[keep]
            v_valid = v_valid[keep]
        else:
            print("[ofs_loader] WARN: land filtering removed all valid FVCOM points — keeping originals")

    valid_tree = cKDTree(valid_pts)
    
    print("[ofs_loader] Built FAST nearest-neighbor interpolator")
    
    def sample(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        
        if lon_0360:
            lons = np.where(lons < 0.0, lons + 360.0, lons)
        
        query_pts = np.column_stack((lons, lats))
        
        # Fast nearest-neighbor lookup
        _, indices = valid_tree.query(query_pts)
        
        u_vals = u_valid[indices]
        v_vals = v_valid[indices]
        return np.column_stack((u_vals, v_vals))
    # annotate sampler for downstream resampling/diagnostics
    sample._native = 'unstructured'
    sample._valid_pts = valid_pts
    sample._u_valid = u_valid
    sample._v_valid = v_valid
    return _attach_source(sample)

    return sample



# def get_current_fn(
#     port: str,
#     start: dt.datetime | None = None,
# ) -> Callable[[np.ndarray, np.ndarray, dt.datetime], np.ndarray]:
#     """Return a callable that samples surface currents in m/s."""
#     if port not in SIMULATION_BOUNDS:
#         raise KeyError(f"Port '{port}' not found in SIMULATION_BOUNDS")

#     cfg = SIMULATION_BOUNDS[port]
#     lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
#     lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))

#     # Inputs may be in UTM – detect & convert once
#     if abs(lon_max) > 180 or abs(lat_max) > 90:
#         utm_zone = int(((lon_min + lon_max) / 2 + 180) // 6) + 1
#         utm_epsg = 32600 + utm_zone
#         to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
#         lon_min, lat_min = to_ll.transform(cfg["minx"], cfg["miny"])
#         lon_max, lat_max = to_ll.transform(cfg["maxx"], cfg["maxy"])
#         lon_min, lon_max = sorted((lon_min, lon_max))
#         lat_min, lat_max = sorted((lat_min, lat_max))

#     bbox = (lon_min, lon_max, lat_min, lat_max)
#     print(f"[ofs_loader] bbox for {port}: "
#           f"({lon_min:.4f}, {lat_min:.4f})–({lon_max:.4f}, {lat_max:.4f})")

#     if start is None:
#         start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)

#     model = OFS_MODEL_MAP.get(port, "rtofs").lower()

#     # # 1) Try the port-specific subset (may be 2D or station-only)
#     # ds = open_ofs_subset(model, start.date(), bbox)
#     # # 2) If that came back as 1D stations, override by directly opening the RTOFS 2-D surface file.
#     # if ds.lon.ndim == 1:
#     #     ymd = start.strftime("%Y%m%d")
#     #     # NOAA S3 key pattern for RTOFS 2DS surface files:
#     #     key = f"rtofs/netcdf/{start.year:04d}/{start.month:02d}/{start.day:02d}/rtofs.t{start:%Hz}z.global.2ds.n000.nc"
#     #     url = f"s3://noaa-nos-ofs-pds/{key}"
#     #     print(f"[ofs_loader] 1D station subset—forcing 2D RTOFS load from {url}")
#     #     ds = xr.open_dataset(url, engine="netcdf4")
#     #     # now crop to our port bbox
#     #     ds = ds.sel(lon=slice(bbox[0], bbox[1]),
#     #                 lat=slice(bbox[2], bbox[3]))
    
#     # 1) Try the port-specific subset (will fetch either a 2-D 2ds file or a 1-D stations file)
#     ds = open_ofs_subset(model, start.date(), bbox)
#     # 2) If the subset is 1-D, *try* RTOFS; keep stations if RTOFS missing
#     if ds["u"].ndim < 3 or ds.lon.ndim == 1:
        
#         try:
#             ds = open_ofs_subset("rtofs", start.date(), bbox)
#             print("[ofs_loader] station subset → switched to RTOFS 2-D grid")
#         except FileNotFoundError:
#             print("[ofs_loader] RTOFS 2-D grid not available – using station data")
        
#     # extract time axis
#     times = ds.time.values

#     # ─── if 2-D grid → true bilinear interp; else 1-D → KD-tree nearest-neighbor ───
#     if ds.lon.ndim == 2:
#         # Structured 2-D grid: interpolate at each query point
#         def sample(lons: np.ndarray, lats: np.ndarray, when: dt):
#             arr = ds.interp(
#                 time=np.datetime64(when),
#                 lon = ("points", lons),
#                 lat = ("points", lats),
#                 method="linear",
#                 kwargs={"fill_value": np.nan},
#             )
#             u = arr.u.values
#             v = arr.v.values
#             # drop the singleton time-dimension if present
#             if u.ndim == 2:
#                 u = u[0]
#                 v = v[0]
#             return np.column_stack((u, v))
#     else:
#         # Unstructured FVCOM-style grid: use scattered interpolator
#         from scipy.interpolate import LinearNDInterpolator
#         from scipy.interpolate import NearestNDInterpolator
    
#         pts = np.column_stack((ds.lon.values, ds.lat.values))
#         if "siglay" in ds["u"].dims:
#             u_flat = ds["u"].isel(siglay=-1, time=0).values
#             v_flat = ds["v"].isel(siglay=-1, time=0).values
#         else:
#             u_flat = ds["u"][0].values
#             v_flat = ds["v"][0].values
            
#         print("[ofs_loader] u_flat stats:", np.nanmin(u_flat), "to", np.nanmax(u_flat))
#         print("[ofs_loader] v_flat stats:", np.nanmin(v_flat), "to", np.nanmax(v_flat))
#         print("[ofs_loader] u_flat shape:", u_flat.shape)
#         print("[ofs_loader] sample lon/lat range:",
#               np.min(ds.lon.values), "to", np.max(ds.lon.values),
#               np.min(ds.lat.values), "to", np.max(ds.lat.values))

#         interp_u_linear = LinearNDInterpolator(pts, u_flat)
#         interp_v_linear = LinearNDInterpolator(pts, v_flat)
#         interp_u_nearest = NearestNDInterpolator(pts, u_flat)
#         interp_v_nearest = NearestNDInterpolator(pts, v_flat)
    
#         def sample(lons: np.ndarray, lats: np.ndarray, when: dt):
#             lons = np.asarray(lons)
#             lons = np.where(lons < 0, lons + 360, lons)  # Convert to 0–360
#             u_vals = interp_u_linear(lons, lats)
#             v_vals = interp_v_linear(lons, lats)
        
#             mask_u = np.isnan(u_vals)
#             mask_v = np.isnan(v_vals)
#             if np.any(mask_u):
#                 u_vals[mask_u] = interp_u_nearest(lons[mask_u], lats[mask_u])
#             if np.any(mask_v):
#                 v_vals[mask_v] = interp_v_nearest(lons[mask_v], lats[mask_v])
        
#             return np.column_stack((u_vals, v_vals))

    
#         return sample


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
    url = None
    for day in (start - timedelta(n) for n in range(0, 15)):
        url = first_existing_url(candidate_met_urls(model, day))
        if url:
            break
    else:
        raise FileNotFoundError(f"No {model.upper()} MET data found in last 14 days")

    print(f"[ofs_loader] -> opening wind {url}")
    # drop the problematic 'siglay' variable at load time
    ds_w = xr.open_dataset(
        fs.open(url),
        engine="h5netcdf",
        chunks={"time": 1},
        drop_variables=["siglay", "siglev"],  # drop vars whose names collide with dims
    )
    
    # Find coordinate names - could be lon/lat, lonc/latc, lon_rho/lat_rho, x/y, etc.
    lon_coord = None
    lat_coord = None
    for possible_lon in ["lon", "lonc", "lon_rho", "longitude", "x"]:
        if possible_lon in ds_w.variables:
            lon_coord = possible_lon
            break
    for possible_lat in ["lat", "latc", "lat_rho", "latitude", "y"]:
        if possible_lat in ds_w.variables:
            lat_coord = possible_lat
            break
    
    if lon_coord is None or lat_coord is None:
        raise ValueError(f"Cannot find lon/lat coordinates in {url}. Available: {list(ds_w.variables)}")
    
    print(f"[ofs_loader] Found coordinates: {lon_coord}, {lat_coord}")
    
    # Rename to standard lon/lat if needed
    if lon_coord != "lon" or lat_coord != "lat":
        rename_dict = {}
        if lon_coord != "lon":
            rename_dict[lon_coord] = "lon"
        if lat_coord != "lat":
            rename_dict[lat_coord] = "lat"
        ds_w = ds_w.rename(rename_dict)
        print(f"[ofs_loader] Renamed {rename_dict} -> lon/lat")
    
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
    
    """
    Return a fast (u,v) sampler using HRRR→ERA5 fallback.
    Keeps the original signature so *all* caller code stays unchanged.
    """
    if port not in SIMULATION_BOUNDS:
        raise KeyError(f"Port '{port}' not found")
    if start is None:
        start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
    lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))
    bbox = (lon_min, lon_max, lat_min, lat_max)

    # delegate to the helpers we just added to atmospheric.py
    from emergent.ship_abm import atmospheric as _atm
    return _atm.wind_sampler(bbox, start)    
    
    
    # if port not in SIMULATION_BOUNDS:
    #     raise KeyError(f"Port '{port}' not found")
    # if start is None:
    #     start = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # model = OFS_MODEL_MAP.get(port, "rtofs").lower()
    # # reuse the same bbox logic as get_current_fn
    # cfg = SIMULATION_BOUNDS[port]
    # lon_min, lon_max = sorted((cfg["minx"], cfg["maxx"]))
    # lat_min, lat_max = sorted((cfg["miny"], cfg["maxy"]))
    # bbox = (lon_min, lon_max, lat_min, lat_max)

    # # --- 1) Coastal stations (.stations.nowcast) sampler ---
    # ds_w = open_met_subset(model, start.date(), bbox)
    # if ds_w.lon.ndim == 1:
    #     # --- 2) FALLBACK TO ATMOSPHERIC GRID (GFS 10 m winds) ----------
    #     from emergent.ship_abm.atmospheric import fetch_gfs_wind_10m
    #     print("[ofs_loader] station-only wind found → fetching GFS 10 m wind grid")
    #     # fetch_gfs_wind_10m returns an xarray with dims (time, lat, lon) and vars 'u10','v10'
    #     ds_w = fetch_gfs_wind_10m(bbox, start)

    # # ─── if we have a 2-D structured grid, do true bilinear interp ─────────
    # if ds_w.lon.ndim == 2:
    #     def sample_wind(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
    #         arr = ds_w.interp(
    #             time = np.datetime64(when),
    #             lon  = ("points", lons),
    #             lat  = ("points", lats),
    #             method = "linear",
    #             kwargs = {"fill_value": np.nan}
    #         )
    #         # arr.u, arr.v have dims ("time","points") → take the first (only) time
    #         uvals = arr.u.values
    #         vvals = arr.v.values
    #         if uvals.ndim == 2:
    #             uvals = uvals[0]
    #             vvals = vvals[0]
    #         return np.column_stack((uvals, vvals))
    # else:
    #     # ─── unstructured (station-based) → KD-tree nearest-neighbour ─────────
    #     times_w = ds_w.time.values
    #     lon_pts = ds_w.lon.values.ravel()
    #     lat_pts = ds_w.lat.values.ravel()
    #     tree_w  = cKDTree(np.column_stack((lon_pts, lat_pts)))
    #     u_flat  = ds_w["u"].values.reshape(ds_w.dims["time"], -1)
    #     v_flat  = ds_w["v"].values.reshape(ds_w.dims["time"], -1)

    #     def sample_wind(lons: np.ndarray, lats: np.ndarray, when: dt.datetime):
    #         idx_t = np.abs(times_w - np.datetime64(when)).argmin()
    #         idx_n = tree_w.query(np.column_stack((lons, lats)), k=1)[1]
    #         return np.column_stack((u_flat[idx_t, idx_n],
    #                                  v_flat[idx_t, idx_n]))

    # return sample_wind
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
