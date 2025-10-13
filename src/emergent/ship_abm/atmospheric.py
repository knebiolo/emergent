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
    # Diagnostic: print u10/v10 shapes and dataset coordinate shapes
    try:
        print(f"[wind_sampler][diag] u10.shape={u_vals.shape} v10.shape={v_vals.shape}")
        coord_info = {name: getattr(coord, 'values', None).shape if getattr(coord, 'values', None) is not None else None for name, coord in ds.coords.items()}
        print(f"[wind_sampler][diag] ds.coords shapes: {coord_info}")
        print(f"[wind_sampler][diag] ds.u10.dims={ds.u10.dims}")
    except Exception:
        pass
    # print("u10:", u_vals)
    # print("v10:", v_vals)

    # Determine which dimension represents time. Datasets vary: some use
    # 'time', others use 'ocean_time' etc. We pick the most likely candidate
    # (a coord with 'time' in its name and length>1). Then normalize u_vals
    # to shape (time, spatial...) for later flattening.
    dims = ds.u10.dims
    # candidate time dims: prefer any dim whose coord name contains 'time' and has length>1
    time_dim = None
    for d in dims:
        if d in ds.coords and 'time' in d.lower() and getattr(ds.coords[d], 'size', 0) > 1:
            time_dim = d
            break
    # fallback to 'time' coord if present
    if time_dim is None and 'time' in ds.coords and getattr(ds.coords['time'], 'size', 0) > 1:
        time_dim = 'time'
    # final fallback: first dim
    if time_dim is None:
        time_dim = dims[0]

    # build t_axis from the chosen time_dim if it contains datetimes
    t_axis = None
    try:
        t_axis = ds.coords[time_dim].values.astype('datetime64[s]').astype(np.int64)
    except Exception:
        try:
            t_axis = ds.coords[time_dim].values
        except Exception:
            t_axis = None

    # Ensure u_vals has a leading time axis at position 0 matching time_dim
    # Move the time_dim to axis 0 then reshape spatial dims later.
    if dims[0] != time_dim:
        # move axis
        axis_idx = dims.index(time_dim)
        u_vals = np.moveaxis(u_vals, axis_idx, 0)
        v_vals = np.moveaxis(v_vals, axis_idx, 0)
        dims = (time_dim,) + tuple(d for d in dims if d != time_dim)

    # ─────────────────────────────────────────────────────────────
    # KDTree Interpolation for Unstructured 1D station data
    # Only use KDTree when the dataset is truly unstructured (lon/lat are 1-D
    # station/node coordinates) or explicitly labeled as 'station'/'node'.
    # If lon/lat are 2-D arrays (structured curvilinear grid), fall through
    # to the structured-grid interpolator below.
    if ("station" in ds.dims) or ("node" in ds.dims) or ("lon" in ds and ds.lon.ndim == 1):
        print("[wind_sampler] using KDTree nearest-neighbor sampling (stations)")

        # Get dimension name aligned with u10 shape
        spatial_dim = ds.u10.dims[-1]

        # Attempt to find coordinates that align with the spatial shape of u_vals
        lon = None
        lat = None
        spatial_shape = u_vals.shape[1:]
        # Look for coords attached to u10 that have the same spatial shape
        for name, coord in getattr(ds.u10, 'coords', {}).items():
            try:
                if hasattr(coord.values, 'shape') and coord.values.shape == spatial_shape:
                    if 'lon' in name.lower() or 'x' in name.lower():
                        lon = coord
                    if 'lat' in name.lower() or 'y' in name.lower():
                        lat = coord
            except Exception:
                continue

        # If we didn't find explicit lon/lat coords attached to u10, try the dataset-level lon/lat
        if lon is None or lat is None:
            if 'lon' in ds and 'lat' in ds:
                # accept dataset-level lon/lat even if their shapes don't exactly
                # match the u10 spatial dims — we'll try to align/transposed below.
                lon = ds.lon
                lat = ds.lat
            else:
                # last resort: try any coords in dataset matching spatial shape
                for name, coord in ds.coords.items():
                    try:
                        if hasattr(coord.values, 'shape') and coord.values.shape == spatial_shape:
                            if lon is None:
                                lon = coord
                            elif lat is None:
                                lat = coord
                    except Exception:
                        continue

        if lon is None or lat is None:
            raise ValueError(f"'u10' variable lacks aligned lon/lat coordinates. Tried spatial_shape={spatial_shape}")

        # Flatten spatial dims (all dims after the leading time_dim) into a
        # single spatial axis. After the moveaxis above, u_vals shape is
        # (time, spatial_dim1, spatial_dim2, ...). We flatten the trailing
        # axes into a single (time, npoints) array.
        if u_vals.ndim >= 2:
            u_flat = u_vals.reshape((u_vals.shape[0], -1))
            v_flat = v_vals.reshape((v_vals.shape[0], -1))
        else:
            u_flat = u_vals.reshape((u_vals.shape[0], -1))
            v_flat = v_vals.reshape((v_vals.shape[0], -1))

        # Align lon/lat to flattened u/v. lon/lat may be 1-D (separable axes)
        # or 2-D (curvilinear). Create flattened coordinate pairs accordingly.
        # Align lon/lat to flattened u/v. lon/lat may be 1-D (separable axes),
        # 2-D (curvilinear grid), or transposed relative to u/v. Handle common
        # cases by transposing when necessary.
        lv = getattr(lon, 'values', None)
        la = getattr(lat, 'values', None)
        if lv is None or la is None:
            raise ValueError("lon/lat coordinates have no .values")

        if lv.ndim == 1 and la.ndim == 1:
            # If lon/lat lengths match the flattened u/v spatial length, they
            # represent station coordinate pairs. If their product matches the
            # flattened length, they are separable axes (meshgrid). Otherwise
            # prefer treating them as station pairs when lon.size == lat.size.
            nspat = u_flat.shape[1]
            if lv.size == nspat:
                lon_flat = lv
                lat_flat = la
            elif lv.size * la.size == nspat:
                lon2d, lat2d = np.meshgrid(lv, la)
                lon_flat = lon2d.ravel()
                lat_flat = lat2d.ravel()
            elif lv.size == la.size:
                lon_flat = lv
                lat_flat = la
            else:
                # fallback: ravel both and hope for the best
                lon_flat = lv.ravel()
                lat_flat = la.ravel()
        elif lv.ndim == 2 and lv.shape == spatial_shape:
            lon_flat = lv.ravel()
            lat_flat = la.ravel()
        elif lv.ndim == 2 and lv.T.shape == spatial_shape:
            lon_flat = lv.T.ravel()
            lat_flat = la.T.ravel()
        else:
            # Last resort: just ravel whatever we have and hope it aligns
            lon_flat = lv.ravel()
            lat_flat = la.ravel()

        # Defensive check: number of flattened spatial points should match the flattened u/v arrays
        if lon_flat.size != u_flat.shape[1]:
            raise ValueError(f"Mismatch: lon size = {lon_flat.size}, u10 flattened shape = {u_flat.shape}")

        xy = np.column_stack((lon_flat, lat_flat))
        tree = cKDTree(xy)
        # Diagnostic: print basic stats about the native u/v arrays so we can
        # detect empty or all-zero data early. This helps debug station-based
        # OFS files that end up producing zero winds in the viewer.
        try:
            u0 = u_flat[0]
            v0 = v_flat[0]
            print(f"[wind_sampler][debug] native u: shape={u_vals.shape} min={np.nanmin(u0):.6f} max={np.nanmax(u0):.6f} nonzero={int(np.count_nonzero(u0))}")
            print(f"[wind_sampler][debug] native v: shape={v_vals.shape} min={np.nanmin(v0):.6f} max={np.nanmax(v0):.6f} nonzero={int(np.count_nonzero(v0))}")
            print(f"[wind_sampler][debug] sample lon/lat pts (first 5): {xy[:5].tolist()}")
        except Exception:
            # non-fatal; continue
            pass

        warned = {"once": False}

        def sample(lon, lat, when):
            t_idx = 0
            if t_axis is not None:
                t_val = np.int64(np.datetime64(when, "s"))
                t_idx = np.abs(t_axis - t_val).argmin()

            u_slice = u_flat[t_idx]
            v_slice = v_flat[t_idx]

            _, idx = tree.query(np.column_stack((lon, lat)), k=1)
            out = np.column_stack((u_slice[idx], v_slice[idx]))

            # One-time warning if this sampler is returning all zeros — helps
            # distinguish between genuinely calm conditions and a loader bug.
            if not warned["once"]:
                if np.allclose(out, 0.0):
                    warned["once"] = True
                    try:
                        sample_idx = idx[:10]
                        print(f"[wind_sampler][warn] sampler returned all zeros for t_idx={t_idx}.\n first_query_idxs={sample_idx.tolist()}\n first_u={u_slice[sample_idx].tolist()}\n first_v={v_slice[sample_idx].tolist()}")
                    except Exception:
                        print(f"[wind_sampler][warn] sampler returned all zeros for t_idx={t_idx} (could not print samples)")

            return out

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
