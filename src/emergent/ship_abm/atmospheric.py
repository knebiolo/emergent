# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:36:41 2025

@author: Kevin.Nebiolo
"""

# atmospheric/hrrr.py
from __future__ import annotations
import datetime as dt, fsspec, xarray as xr
from typing import Tuple
import numpy as np, xarray as xr, datetime as dt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# ---------- HRRR (3 km CONUS, 2014-present, hourly) ----------------
_HRRR_BUCKET = "noaa-hrrr-pds"
_hrrr_fs     = fsspec.filesystem("s3", anon=True)

# HRRR path pattern:
#   hrrr.YYYYMMDD/conus/hrrr.tCCz.wrfprsfFFF.grib2
#   where CC = cycle hour (00…23)  •  FFF = forecast hour 000…018
def _hrrr_key(when: dt.datetime) -> str:
    ymd = when.strftime("%Y%m%d")
    cyc = f"{when:%H}"        # "14", no trailing "Z"
    fxx = "000"               # we almost always want the analysis (f00)
    return f"hrrr.{ymd}/conus/hrrr.t{cyc}z.wrfprsf{fxx}.grib2"

def fetch_hrrr_wind_10m(bbox: Tuple[float, float, float, float],
                        when: dt.datetime) -> xr.Dataset:
    """Return 2-hour slab (time, lat, lon) with u10/v10 as float32."""
    url = f"s3://{_HRRR_BUCKET}/{_hrrr_key(when)}"
    if not _hrrr_fs.exists(url[5:]):           # strip “s3://”
        raise FileNotFoundError("HRRR slice absent")
    ds = xr.open_dataset(
        _hrrr_fs.open(url), engine="cfgrib",
        backend_kwargs={"filter_by_keys":
                        {"typeOfLevel": "heightAboveGround", "level": 10}}
    ).rename({"u": "u10", "v": "v10"})
    ds = ds.sel(longitude=slice(bbox[0], bbox[1]),
                latitude =slice(bbox[3], bbox[2]))          # HRRR lat desc
    return (ds.rename({"longitude": "lon", "latitude": "lat"})
              .isel(time=slice(0, 2))            # this hour + next
              .load()
              .astype(np.float32))

# ---------- ERA5 (31 km global, 1940-present, hourly) --------------
_ERA_BUCKET = "era5-pds"
_era_fs     = fsspec.filesystem("s3", anon=True)

def _era_key(var: str, when: dt.datetime) -> str:
    return f"{when:%Y}/{when:%m}/data/{var}/{var}_{when:%Y%m%d}.nc"

def fetch_era5_wind_10m(bbox: Tuple[float, float, float, float],
                        when: dt.datetime) -> xr.Dataset:
    files = []
    for var in ("u10", "v10"):
        url = f"s3://{_ERA_BUCKET}/{_era_key(var, when)}"
        try:
            files.append(xr.open_dataset(_era_fs.open(url),
                                         engine="h5netcdf", chunks={}))
        except FileNotFoundError:
            raise FileNotFoundError("ERA5 slice absent")
    ds = xr.merge(files, combine_attrs="override")
    ds = ds.sel(longitude=slice(bbox[0], bbox[1]),
                latitude =slice(bbox[3], bbox[2]),
                time     =[when, when + dt.timedelta(hours=1)])
    return (ds.rename({"longitude": "lon", "latitude": "lat"})
              .load()
              .astype(np.float32))

def fetch_noaa_ofs_wind(bbox: Tuple[float, float, float, float],
                        when: dt.datetime) -> xr.Dataset:
    """
    Fallback loader using NOAA OFS .met.nowcast files from coastal models.
    Looks for wind variables (e.g., 'Uwind', 'Vwind', 'eastward_wind') and reshapes as needed.
    """
    from emergent.ship_abm.ofs_loader import open_met_subset, OFS_MODEL_MAP, SIMULATION_BOUNDS

    # Pick appropriate model from port bounding box
    port_bbox_matches = [k for k, v in SIMULATION_BOUNDS.items()
                         if (v["minx"] <= bbox[0] <= v["maxx"]) and (v["miny"] <= bbox[2] <= v["maxy"])]
    if port_bbox_matches:
        model = OFS_MODEL_MAP.get(port_bbox_matches[0], "rtofs")
    else:
        model = "rtofs"

    try:
        ds = open_met_subset(model, when.date(), bbox)
    except Exception as e:
        raise FileNotFoundError("NOAA OFS wind not available") from e

    # Guess wind variable names
    var_u = next((v for v in ds.data_vars if "eastward" in ds[v].attrs.get("standard_name", "").lower()
                  or "uwind" in v.lower()), None)
    var_v = next((v for v in ds.data_vars if "northward" in ds[v].attrs.get("standard_name", "").lower()
                  or "vwind" in v.lower()), None)

    if not (var_u and var_v):
        raise KeyError(f"No wind variables found in coastal MET file. Found: {list(ds.data_vars)}")

    ds = ds.rename({var_u: "u10", var_v: "v10"})
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
        ds["time"] = [np.datetime64(when)]

    return ds.astype(np.float32)




# ---------- High-speed tri-linear sampler -------------------------
from scipy.spatial import cKDTree

def build_wind_sampler(ds: xr.Dataset):
    """
    Compile a vectorised (lon, lat, when) → (u10, v10) callable.

    Automatically chooses interpolation method:
      - RegularGridInterpolator for structured (grid) data
      - KDTree nearest-neighbor for unstructured (station) data
    """

    u_vals = ds.u10.values
    v_vals = ds.v10.values
    # print("u10:", u_vals)
    # print("v10:", v_vals)

    # Standardize to 3D: (time, y, x) or (time, station)
    if u_vals.ndim == 2 and "time" in ds.u10.dims:  # (time, station)
        t_axis = ds.time.values.astype("datetime64[s]").astype(np.int64)
    elif u_vals.ndim == 2:  # (lat, lon)
        u_vals = u_vals[None, ...]
        v_vals = v_vals[None, ...]
        t_axis = np.array([np.datetime64(ds.time.values[0], "s").astype(np.int64)])
    elif u_vals.ndim == 3:  # (time, lat, lon)
        t_axis = ds.time.values.astype("datetime64[s]").astype(np.int64)
    else:
        raise ValueError(f"Unsupported wind array shape: {u_vals.shape}")

    # ─────────────────────────────────────────────────────────────
    # KDTree Interpolation for Unstructured 1D station data
    # ─────────────────────────────────────────────────────────────
    if "station" in ds.dims or "node" in ds.dims or u_vals.shape[-1] == ds.lon.size:
        print("[wind_sampler] using KDTree nearest-neighbor sampling (stations)")

        # Get dimension name aligned with u10 shape
        spatial_dim = ds.u10.dims[-1]

        try:
            # Grab properly aligned coordinates from the u10 variable itself
            lon = ds.u10.coords["lon"]
            lat = ds.u10.coords["lat"]
            # print("LON:", lon)
            # print("LAT:", lat)
        except KeyError:
            raise ValueError(f"'u10' variable lacks aligned 'lon'/'lat' coordinates.")

        # Defensive check
        if lon.size != u_vals.shape[-1]:
            raise ValueError(f"Mismatch: lon size = {lon.size}, u10 shape = {u_vals.shape}")

        xy = np.column_stack((lon.values, lat.values))
        tree = cKDTree(xy)

        def sample(lon, lat, when):
            t_idx = 0
            if t_axis is not None:
                t_val = np.int64(np.datetime64(when, "s"))
                t_idx = np.abs(t_axis - t_val).argmin()

            u_slice = u_vals[t_idx]
            v_slice = v_vals[t_idx]

            _, idx = tree.query(np.column_stack((lon, lat)), k=1)
            return np.column_stack((u_slice[idx], v_slice[idx]))

        return sample

    # ─────────────────────────────────────────────────────────────
    # Structured Grid Interpolation using RegularGridInterpolator
    # ─────────────────────────────────────────────────────────────
    print("[wind_sampler] using RegularGridInterpolator (structured grid)")

    lats = ds.lat.values
    lons = ds.lon.values
    
    # print("LON:", lons)
    # print("LAT:", lats)

    # Sort lats
    if not (np.all(np.diff(lats) > 0) or np.all(np.diff(lats) < 0)):
        lat_sort = np.argsort(lats)
        lats = lats[lat_sort]
        u_vals = u_vals[:, lat_sort, :]
        v_vals = v_vals[:, lat_sort, :]

    # Sort lons
    if not (np.all(np.diff(lons) > 0) or np.all(np.diff(lons) < 0)):
        lon_sort = np.argsort(lons)
        lons = lons[lon_sort]
        u_vals = u_vals[:, :, lon_sort]
        v_vals = v_vals[:, :, lon_sort]

    u_fun = RegularGridInterpolator((t_axis, lats, lons), u_vals,
                                    bounds_error=False, fill_value=np.nan)
    v_fun = RegularGridInterpolator((t_axis, lats, lons), v_vals,
                                    bounds_error=False, fill_value=np.nan)

    def sample(lon: np.ndarray, lat: np.ndarray, when: dt.datetime):
        t_int = np.int64(np.datetime64(when, "s"))
        pts   = np.column_stack((np.full_like(lon, t_int), lat, lon))
        return np.column_stack((u_fun(pts), v_fun(pts)))

    return sample

# ---------- One-liner your ABM can import -------------------------
def wind_sampler(bbox: Tuple[float, float, float, float],
                 start: dt.datetime):
    """
    Convenience wrapper: try HRRR first, fall back to ERA5, then NOAA OFS.
    Returns a function f(lon, lat, when) → (u, v).
    """
    try:
        ds = fetch_hrrr_wind_10m(bbox, start)
    except FileNotFoundError:
        try:
            ds = fetch_era5_wind_10m(bbox, start)
        except (FileNotFoundError, PermissionError):
            print("[wind_sampler] HRRR and ERA5 failed — falling back to NOAA OFS")
            ds = fetch_noaa_ofs_wind(bbox, start)

    return build_wind_sampler(ds)
