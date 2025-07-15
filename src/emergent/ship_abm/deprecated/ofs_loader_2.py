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
from typing import Callable, Iterable
import datetime as dt

import fsspec
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree  
# Single source-of-truth for bounds + model names
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP
from pyproj import Transformer
from datetime import date                      # <-- NEW: gives plain “date” name

#LAYERS = ("n000", "f000")                 # add f003 … f048 if you ever need them

# --------------------------------------------------------------------------- #
# constants & helpers
# --------------------------------------------------------------------------- #
# NOAA publishes to both buckets for now; probe the NOS one first.
BUCKETS: tuple[str, ...] = ("noaa-nos-ofs-pds", "noaa-ofs-pds")
# operational cycles (3-hourly) — try most-recent first
CYCLES:  tuple[int, ...] = (18, 15, 12, 9, 6, 3, 0, 21)


#fs = fsspec.filesystem("s3", anon=True)

def regional_keys(model: str, day: date) -> Iterable[str]:
    """
    Yield every candidate *key* (no bucket prefix) that might hold 2-D surface
    currents for *model* on *day*.
    """
    y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
    ymd     = f"{y}{m}{d}"
    layers  = ("n000", "f000")                      # nowcast | forecast
    for cyc in CYCLES:
        for layer in layers:
            tmpl = f"{model}.t{cyc:02d}z.{ymd}.2ds.{layer}.nc"
            # canonical layout
            yield f"{model}/netcdf/{y}/{m}/{d}/{tmpl}"
            # transitional 2024- legacy layout
            yield f"{model}/netcdf/{y}{m}/{tmpl}"

def candidate_urls(model: str, day: date) -> list[str]:
    """Return fully-qualified s3:// URLs for every bucket + key combo."""
    keys = list(regional_keys(model, day))
    return [f"s3://{bucket}/{key}"
            for bucket in BUCKETS
            for key    in keys]

def first_existing_url(urls: list[str]) -> str | None:
    for url in urls:
        bucket, key = url.removeprefix("s3://").split("/", 1)
        if fs.exists(f"{bucket}/{key}"):
            return url
    return None
    
def _open_ofs_subset(
    model: str,
    start: dt.date,
    bbox: tuple[float, float, float, float],
    vars: tuple[str, str] = ("ua", "va"),
    lookback: int = 14,                       # days to search backwards
) -> xr.Dataset:
    """
    Grab the newest OFS file within *lookback* days, open it with xarray,
    pick a usable (u,v) pair, and crop to *bbox*.

    Parameters
    ----------
    model   : e.g. ``"ngofs2"``  (case-insensitive)
    start   : first date to try (usually today)
    bbox    : (lon_min, lon_max, lat_min, lat_max)
    vars    : preferred variable names in the file
    """
    fs = fsspec.filesystem("s3", anon=True)   # ← **BUG FIX**  you had this #
                                              #    -commented

    # -----------------------------------------------------------------
    # 1.  Assemble candidate URLs -------------------------------------
    # -----------------------------------------------------------------
    # CYCLES = (18, 15, 12, 9, 6, 3, 0, 21)
    # BUCKETS = ("noaa-nos-ofs-pds", "noaa-ofs-pds")

    def _keys(day: dt.date) -> Iterable[str]:
        y, m, d = day.strftime("%Y"), day.strftime("%m"), day.strftime("%d")
        ymd = f"{y}{m}{d}"
        for cyc in CYCLES:
            for tmpl in (
                f"{model}.t{cyc:02d}z.{ymd}.2ds.n000.nc",
                f"{model}.t{cyc:02d}z.{ymd}.2ds.f000.nc",
                f"nos.{model}.nowcast.{ymd}.t{cyc:02d}z.nc",     # legacy
            ):
                yield f"{model}/netcdf/{y}/{m}/{d}/{tmpl}"
                yield f"{model}/netcdf/{y}{m}/{tmpl}"
                yield f"{model.upper()}.{ymd}/{tmpl}"

    def _urls(day: dt.date) -> list[str]:
        return [f"s3://{b}/{k}" for k in _keys(day) for b in BUCKETS]

    # walk back until we hit something that exists
    url = None
    for delta in range(lookback):
        for u in _urls(start - dt.timedelta(days=delta)):
            bucket, key = u.replace("s3://", "", 1).split("/", 1)
            if fs.exists(f"{bucket}/{key}"):
                url = u
                break
        if url:
            break
    if url is None:
        raise FileNotFoundError(f"No {model.upper()} data in last {lookback} days")

    print(f"[ofs_loader] → opening   {url}")
    ds = xr.open_dataset(fs.open(url), engine="h5netcdf", chunks={"time": 1})

    # -----------------------------------------------------------------
    # 2.  Choose a (u,v) variable pair --------------------------------
    # -----------------------------------------------------------------
    var_pairs = [
        vars, ("us", "vs"), ("u", "v"), ("water_u", "water_v")
    ]

    # add any CF-compliant names the file might advertise
    east  = [v for v in ds.data_vars if "eastward"  in ds[v].attrs.get("standard_name","")]
    north = [v for v in ds.data_vars if "northward" in ds[v].attrs.get("standard_name","")]
    var_pairs.extend(zip(east, north))

    for u, v in var_pairs:
        if u in ds and v in ds:
            ds = ds[[u, v]].rename({u: "u", v: "v"})
            break
    else:
        raise KeyError(f"No current-vector variables in {url!s}")

    # -----------------------------------------------------------------
    # 3.  Normalise lon/lat names & crop ------------------------------
    # -----------------------------------------------------------------
    for lo, la in (("lon","lat"), ("lonc","latc"), ("longitude","latitude")):
        if lo in ds and la in ds:
            ds = ds.rename({lo: "lon", la: "lat"})
            break

    # promote lonc/latc (FVCOM) to coordinates
    if {"lonc","latc"} <= set(ds.data_vars):
        ds = ds.set_coords(["lonc", "latc"])

    lon0, lon1, lat0, lat1 = bbox
    structured = "lon" in ds.coords and ds["lon"].ndim == 2
    if structured:
        mask = (
            (ds.lon > lon0) & (ds.lon < lon1) &
            (ds.lat > lat0) & (ds.lat < lat1)
        )
    elif {"lonc","latc"} <= set(ds.coords):
        mask = (
            (ds.lonc > lon0) & (ds.lonc < lon1) &
            (ds.latc > lat0) & (ds.latc < lat1)
        )
    else:
        mask = None  # unstructured grid without lon/lat info

    if mask is not None:
        ds = ds.where(mask, drop=True)

    print(f"[ofs_loader] ✔ opened    {url}")
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
