# -*- coding: utf-8 -*-
"""
emergent.ship_abm.ofs_loader
----------------------------
Vectorised helper that returns a (u,v) sampler for any port defined in
config.SIMULATION_BOUNDS.  Falls back to global RTOFS if the regional grid
is missing or you forget to map a port.

No external state: you can call get_current_fn() in unit-tests without touching
the rest of the ABM.
"""
from __future__ import annotations
from typing import Callable, Tuple, List, Iterable
import datetime as dt

import fsspec, scipy
import numpy as np
import xarray as xr
import itertools
from itertools import product
from scipy.spatial import cKDTree  
# Single source-of-truth for bounds + model names
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP
from pyproj import Transformer
import math
from datetime import date, timedelta                      # <-- NEW: gives plain “date” name

BUCKETS_REG = (
    "noaa-ofs-pds",        # NOMADS (30-day rolling)
    "noaa-nos-ofs-pds",    # CO-OPS ops & history
)

# --------------------------------------------------------------------------- #
# low-level S3 reader (unchanged)
# --------------------------------------------------------------------------- #
def _open_ofs_subset(
    model: str,
    start: dt.datetime,
    bbox: Tuple[float, float, float, float],
    vars: Tuple[str, str] = ("u", "v"),
) -> xr.Dataset:
    """
    Find the newest NOS-OFS cycle ≤14 days old, open it with xarray, and
    (for structured grids) crop to *bbox*.
    """
    fs = fsspec.filesystem("s3", anon=True)

    # ----------------------------------------------------------------------
    BUCKETS = ("noaa-nos-ofs-pds", "noaa-ofs-pds")       # NOS first
    CYCLES  = (18, 15, 12, 9, 6, 3, 0, 21)

    # ---------- key builders ---------------------------------------------
    def regional_keys(model: str, day: date) -> Iterable[str]:
        y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
        ymd     = f"{y}{m}{d}"
        for cyc in CYCLES:
            for tmpl in (
                f"{model}.t{cyc:02d}z.{ymd}.2ds.n000.nc",    # now-cast,  2-D
                f"{model}.t{cyc:02d}z.{ymd}.2ds.f000.nc",    # forecast, 2-D
            ):
                yield f"{model}/netcdf/{y}/{m}/{d}/{tmpl}"   # new YYYY/MM/DD layout
                yield f"{model}/netcdf/{y}{m}/{tmpl}"        # legacy YYYYMM layout
                yield f"{model.upper()}.{ymd}/{tmpl}"        # snapshot layout

    def candidate_urls(model: str, day: date) -> list[str]:
        """Return fully-qualified s3:// URLs for every bucket + key combo."""
        keys = list(regional_keys(model, day))
        return [f"s3://{bucket}/{key}"
                for bucket in BUCKETS
                for key    in keys]

    # ---------- probe ≤14 days back --------------------------------------
    def first_existing_url(urls: list[str]) -> str | None:
        for url in urls:
            bucket, key = url.removeprefix("s3://").split("/", 1)
            if fs.exists(f"{bucket}/{key}"):
                return url
        return None

    url = None
    for delta in range(0, 14):                           # today .. 13 days ago
        day  = (start - timedelta(days=delta)).date()
        url  = first_existing_url(candidate_urls(model, day))
        if url:
            break

    if url is None:
        raise FileNotFoundError(f"No {model.upper()} data within 14 days.")

    print(f"[ofs_loader] → opening   {url}")
    # `h5netcdf` understands random-access file-like objects, so we can
    # stream straight from S3 without a temp-file download.
    ds = xr.open_dataset(
        fs.open(url, mode="rb"),  # binary read mode
        engine="h5netcdf",
        chunks={"time": 1},       # lazily load one timestep at a time
    )[vars]

    print(f"[ofs_loader] ✔ opened    {url}")

    # ---------- crop (structured grids only) -----------------------------
    lon_min, lon_max, lat_min, lat_max = bbox
    if "node" not in ds.dims:               # unstructured grids (FVCOM) keep all
        ds = ds.where((ds.lon > lon_min) & (ds.lon < lon_max) &
                      (ds.lat > lat_min) & (ds.lat < lat_max),
                      drop=True)
    return ds

# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def get_current_fn(
    port: str,
    start: dt.datetime | None = None,
) -> Callable[[np.ndarray, np.ndarray, dt.datetime], np.ndarray]:
    """
    Parameters
    ----------
    port   : one of the keys in SIMULATION_BOUNDS / OFS_MODEL_MAP
    start  : UTC datetime your simulation *starts* (defaults = now, rounded to
             previous 6-h NOAA cycle)

    Returns
    -------
    sample(x_lon, y_lat, when) → np.ndarray (N,2) with (u,v) in m/s
    """
    if port not in SIMULATION_BOUNDS:
        raise KeyError(
            f"Port '{port}' not found in SIMULATION_BOUNDS.  "
            f"Available keys → {list(SIMULATION_BOUNDS)}"
        )

    # ------------------------------------------------------------------ #
    # 1) Get lon/lat bounds — detect unit automatically
    # ------------------------------------------------------------------ #
    minx = SIMULATION_BOUNDS[port]["minx"]
    maxx = SIMULATION_BOUNDS[port]["maxx"]
    miny = SIMULATION_BOUNDS[port]["miny"]
    maxy = SIMULATION_BOUNDS[port]["maxy"]

    looks_like_degrees = (
        max(abs(minx), abs(maxx)) <= 180
        and max(abs(miny), abs(maxy)) <= 90
    )

    if looks_like_degrees:
        lon_min, lon_max = sorted((minx, maxx))
        lat_min, lat_max = sorted((miny, maxy))
    else:
        # numbers are metres → convert UTM → lon/lat once
        utm_zone = int(((minx + maxx) / 2 + 180) // 6) + 1
        utm_epsg = 32600 + utm_zone
        ll_tx = Transformer.from_crs(
            f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True
        )
        lon_min, lat_min = ll_tx.transform(minx, miny)
        lon_max, lat_max = ll_tx.transform(maxx, maxy)
        lon_min, lon_max = sorted((lon_min, lon_max))
        lat_min, lat_max = sorted((lat_min, lat_max))

    bbox = (lon_min, lon_max, lat_min, lat_max)

    print(f"[ofs_loader] bbox for {port}: "
          f"({lon_min:.4f}, {lat_min:.4f})–({lon_max:.4f}, {lat_max:.4f})")   
    model = OFS_MODEL_MAP.get(port, "rtofs")  # graceful global fallback

    if start is None:
        start = (
            dt.datetime.utcnow()
            .replace(minute=0, second=0, microsecond=0)
        )

    try:
        ds = _open_ofs_subset(model, start, bbox)
    except FileNotFoundError:
        # regional grid missing → always succeed with RTOFS
        try:                                    # global fallback
            ds = _open_ofs_subset("rtofs", start, bbox)
        except Exception:
            # ultimate fallback – zero field so the sim still runs
            def _zero_sample(x, y, t):
                return np.zeros((len(x), 2))
            return _zero_sample
    # ------------------------------------------------------------------ #
    # Detect grid type
    # ------------------------------------------------------------------ #
    is_unstructured = set(ds.dims) >= {"node"}      # all FVCOM estuary grids
    time_coord      = ds.time.values               # 1-D array of datetimes
    
    if is_unstructured:
        # Build a KD-tree once
        lon_nodes = ds["lon"].values
        lat_nodes = ds["lat"].values
        xy_nodes  = np.column_stack((lon_nodes, lat_nodes))
        if cKDTree is not None:
            tree = cKDTree(xy_nodes)
            def _nearest_idx(lon, lat):
                _, idx = tree.query(np.column_stack((lon, lat)), k=1)
                return idx
        else:                                      # pure-NumPy fallback
            def _nearest_idx(lon, lat):
                d2 = (lon[:, None] - lon_nodes)**2 + (lat[:, None] - lat_nodes)**2
                return d2.argmin(axis=1)
    
        # Surface layer (siglay = -1) is enough for ship ABM
        u0 = ds["u"].isel(siglay=-1)               # dims (time,node)
        v0 = ds["v"].isel(siglay=-1)
    
        def sample(lon, lat, when):
            """Return (N,2) np.array of (u,v) in m/s at surface layer."""
            # nearest time step (they’re hourly)
            t_idx = np.abs(time_coord - np.datetime64(when)).argmin()
            n_idx = _nearest_idx(lon, lat)         # (N,)
            return np.column_stack((u0[t_idx, n_idx].values,
                                    v0[t_idx, n_idx].values))
    else:
        # Structured grid (e.g. global RTOFS) → keep xarray.interp path
        def sample(lon, lat, when):
            arr = ds.interp(
                time=np.datetime64(when),
                lon=xr.DataArray(lon, dims="obs"),
                lat=xr.DataArray(lat, dims="obs"),
                method="linear",
                kwargs={"fill_value": np.nan},
            )
            return np.column_stack((arr.u.values, arr.v.values))
    # ################################################################################
    # # Keep the rest of get_current_fn() unchanged
    # ################################################################################
    # # ---------- broadcast-friendly sampler --------------------------------- #
    # def sample(x_lon: np.ndarray, y_lat: np.ndarray, when: dt.datetime):
    #     arr = ds.interp(
    #         time=xr.DataArray([np.datetime64(when)], dims="time"),
    #         lon=xr.DataArray(x_lon, dims="obs"),
    #         lat=xr.DataArray(y_lat, dims="obs"),
    #         method="linear",
    #         kwargs={"fill_value": np.nan},
    #     )
    #     return np.column_stack((arr.u.values[0], arr.v.values[0]))

    return sample
