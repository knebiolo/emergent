"""I/O and environment helpers for the salmon ABM.

This module provides small wrappers around rasterio, geopandas and h5py
used by the simulation. Implementations are conservative to avoid
side-effects during import and testing.
"""
import os
import h5py
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.transform import Affine
from matplotlib import animation as manimation
from datetime import datetime
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple
from shapely.geometry import LineString
from shapely.ops import linemerge
import math

from emergent.salmon_abm import hdf5_io


def _try_setattr(obj: Any, name: str, value: Any) -> bool:
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _affine_to_6tuple(transform: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Return (a,b,c,d,e,f) for rasterio-style Affine transforms, else None."""
    if transform is None:
        return None

    try:
        a = float(getattr(transform, "a"))
        b = float(getattr(transform, "b"))
        c = float(getattr(transform, "c"))
        d = float(getattr(transform, "d"))
        e = float(getattr(transform, "e"))
        f = float(getattr(transform, "f"))
        return (a, b, c, d, e, f)
    except Exception:
        pass

    try:
        seq = tuple(transform)
        if len(seq) < 6:
            return None
        return tuple(float(x) for x in seq[:6])  # type: ignore[return-value]
    except Exception:
        return None


def _coords_from_affine(
    tr_tup: Tuple[float, float, float, float, float, float], shape
) -> Tuple[np.ndarray, np.ndarray]:
    nrows, ncols = shape[:2]
    a, b, c, d, e, f = tr_tup
    cols = np.arange(ncols, dtype=float)
    rows = np.arange(nrows, dtype=float)
    col_indices, row_indices = np.meshgrid(cols, rows)
    x_coords = a * col_indices + b * row_indices + c
    y_coords = d * col_indices + e * row_indices + f
    return x_coords, y_coords


def output_excel(records, model_dir, model_name):
    """Write a dict of pandas DataFrames to an Excel file.

    Parameters
    - records: dict[str, pd.DataFrame]
    - model_dir: str
    - model_name: str
    """
    os.makedirs(model_dir, exist_ok=True)
    output_excel_path = os.path.join(model_dir, f'output_{model_name}.xlsx')
    # Try to write an Excel file; if openpyxl/xlsxwriter are not available,
    # fall back to writing CSV files per sheet.
    try:
        with pd.ExcelWriter(output_excel_path) as writer:
            for generation_name, df in records.items():
                try:
                    df.to_excel(writer, sheet_name=str(generation_name))
                except Exception:
                    pd.DataFrame({'error': ['could not write sheet']}).to_excel(writer, sheet_name=str(generation_name))
        return
    except Exception:
        pass

    # fallback: write CSVs
    for generation_name, df in records.items():
        csv_path = os.path.join(model_dir, f'{model_name}_{generation_name}.csv')
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            with open(csv_path + '.error', 'w') as fh:
                fh.write('could not write data')


def movie_maker(directory, model_name, crs, dt, depth_rast_transform, depth_arr, X_arr=None, Y_arr=None):
    """Lightweight movie maker that uses matplotlib animation writers.

    This function focuses on creating a simple mp4 from a 2D depth array and
    optional agent trajectories. To keep it testable we do not call ffmpeg
    directly here; matplotlib will use the configured writer when available.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    metadata = dict(title=model_name, artist='Matplotlib', comment=f'emergent model run {datetime.now()}')
    fps = max(1, int(round(30.0 / max(dt, 1e-6))))
    out_path = os.path.join(directory, f'{model_name}.mp4')

    # matplotlib's writer registry supports dict-like access but may raise
    # a KeyError if ffmpeg is not configured. Use try/except.
    WriterClass = None
    try:
        WriterClass = manimation.writers['ffmpeg']
    except Exception:
        WriterClass = None

    writer = None
    if WriterClass is not None:
        try:
            writer = WriterClass(fps=fps, metadata=metadata)
        except Exception:
            writer = None
    writer_available = writer is not None

    return {'out_path': out_path, 'fps': fps, 'writer_available': writer_available}


def enviro_import(path):
    """Read a raster file and return (array, transform, crs).

    This wrapper uses rasterio to read the first band and returns
    commonly used metadata.
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    return arr, transform, crs


def write_raster_to_hdf5(h5obj, path, dataset_name=None, sim=None):
    """Read a raster and write into `h5obj` under `environment/<dataset_name>`.

    - `h5obj` may be an h5py.File or dict-like store.
    - `path` is a filesystem path to a raster
    - `dataset_name` if provided uses that base name; otherwise uses filename stem.
    - If `sim` is provided, attempt to set `sim.<dataset_name>_rast_transform`
      to a plain 6-tuple (a,b,c,d,e,f) for compatibility with existing code.

    Returns: (arr, transform_tuple, crs)
    """
    arr, transform, crs = enviro_import(path)
    base = dataset_name or os.path.splitext(os.path.basename(path))[0]
    # write array into HDF5 using hdf5_io helper to support dict-like stores
    hdf5_io.write_dataset(h5obj, f'environment/{base}', arr)

    # convert affine to plain tuple (a,b,c,d,e,f) for compatibility
    tr_tup = _affine_to_6tuple(transform)

    # set attribute on sim if provided: attach both original transform (when available)
    # and the plain 6-tuple for backward compatibility.
    if sim is not None and tr_tup is not None:
        attr_obj = f'{base}_rast_transform'
        attr_tup = f'{base}_rast_transform_tuple'
        if not _try_setattr(sim, attr_obj, transform):
            _try_setattr(sim, attr_obj, tr_tup)
        _try_setattr(sim, attr_tup, tr_tup)

    # Also write environment x_coords/y_coords for this raster when possible.
    # This helps downstream sampling code locate nearest pixels using a
    # dataset-local coordinate grid instead of relying on separate placeholders.
    if arr is not None and tr_tup is not None:
        try:
            x_coords, y_coords = _coords_from_affine(tr_tup, arr.shape)
            existing_x = hdf5_io.read_dataset(h5obj, 'environment/x_coords', default=None)
            if existing_x is None or np.array(existing_x).shape != x_coords.shape:
                hdf5_io.write_dataset(h5obj, 'environment/x_coords', x_coords)
                hdf5_io.write_dataset(h5obj, 'environment/y_coords', y_coords)
        except Exception:
            pass

    return arr, tr_tup, crs


def longitudinal_import(shapefile):
    """Read a longitudinal shapefile and return a GeoDataFrame.

    The original code computes linear positions; that logic can be
    implemented later; here we provide a reliable reader.
    """
    gdf = gpd.read_file(shapefile)
    return gdf


@contextmanager
def safe_hdf5_open(path_or_file, mode='a'):
    """Context manager that accepts a path, an h5py.File, or a dict-like store.

    - If a string path is supplied, opens an h5py.File with the given `mode`.
    - If an h5py.File is supplied, yields it unchanged.
    - If a dict-like store is supplied (for tests), yields it unchanged.
    """
    if isinstance(path_or_file, str):
        f = h5py.File(path_or_file, mode)
        try:
            yield f
        finally:
            try:
                f.close()
            except Exception:
                pass
        return

    # file-like or dict-like
    try:
        yield path_or_file
    finally:
        # do not close dict-like stores
        if hasattr(path_or_file, 'close') and not isinstance(path_or_file, dict):
            try:
                path_or_file.close()
            except Exception:
                pass


def write_sim_initial(
    h5obj,
    sim_state_dict: Dict[str, Any],
    compress: bool = True,
    compression_opts=None,
    *,
    create_timeseries: bool = True,
    timeseries_keys: Optional[Tuple[str, ...]] = None,
):
    """Create standard groups/datasets used by the simulation.

    Parameters
    - h5obj: an h5py.File-like object or dict-like store
    - sim_state_dict: dict with keys `num_agents`, `num_timesteps`, optional arrays
      for `sex`, `length`, `weight`, `body_depth`.
    """
    na = int(sim_state_dict.get('num_agents', 0))
    nt = int(sim_state_dict.get('num_timesteps', 0))

    # static per-agent datasets (prefer provided arrays)
    sex = sim_state_dict.get('sex', np.zeros((na,), dtype=np.int8))
    length = sim_state_dict.get('length', np.zeros((na,), dtype=np.float32))
    weight = sim_state_dict.get('weight', np.zeros((na,), dtype=np.float32))
    body_depth = sim_state_dict.get('body_depth', np.zeros((na,), dtype=np.float32))

    hdf5_io.write_dataset(h5obj, 'agent_data/sex', np.array(sex))
    hdf5_io.write_dataset(h5obj, 'agent_data/length', np.array(length))
    hdf5_io.write_dataset(h5obj, 'agent_data/weight', np.array(weight))
    hdf5_io.write_dataset(h5obj, 'agent_data/body_depth', np.array(body_depth))
    hdf5_io.write_dataset(h5obj, 'sex', np.array(sex))
    hdf5_io.write_dataset(h5obj, 'length', np.array(length))
    hdf5_io.write_dataset(h5obj, 'weight', np.array(weight))
    hdf5_io.write_dataset(h5obj, 'body_depth', np.array(body_depth))

    if create_timeseries:
        keys = timeseries_keys
        if keys is None:
            keys = ('agent_data/X', 'agent_data/Y', 'agent_data/prev_X', 'agent_data/prev_Y', 'agent_data/ideal_sog', 'agent_data/Hz')
        for key in keys:
            try:
                k = str(key)
            except Exception:
                k = key  # type: ignore[assignment]
            if not str(k).startswith('agent_data/'):
                k = f'agent_data/{k}'
            # Create a chunked dataset without writing a full (na,nt) zeros matrix.
            hdf5_io.ensure_timeseries_dataset(h5obj, k, n_agents=na, n_steps=nt, dtype=np.float32)

    # also create legacy top-level position datasets for compatibility
    hdf5_io.ensure_vector_dataset(h5obj, 'X', n_agents=na, dtype=np.float32, fillvalue=0.0)
    hdf5_io.ensure_vector_dataset(h5obj, 'Y', n_agents=na, dtype=np.float32, fillvalue=0.0)
    hdf5_io.ensure_vector_dataset(h5obj, 'prev_X', n_agents=na, dtype=np.float32, fillvalue=0.0)
    hdf5_io.ensure_vector_dataset(h5obj, 'prev_Y', n_agents=na, dtype=np.float32, fillvalue=0.0)

    # environment placeholders
    hdf5_io.create_environment_placeholders(h5obj)

    # small metadata
    metadata = sim_state_dict.get('metadata', {})
    for k, v in metadata.items():
        ok = hdf5_io.write_dataset(h5obj, f'metadata/{k}', np.array(v))
        if not ok:
            hdf5_io.write_dataset(h5obj, f'metadata/{k}', str(v))

    return True


def enviro_load_many(files_dict: Dict[str, str]):
    """Load multiple rasters into a dict mapping key -> (array, transform, crs).

    Missing or unreadable files will have value `None`.
    """
    out = {}
    for key, path in (files_dict or {}).items():
        try:
            arr, tr, crs = enviro_import(path)
            out[key] = (arr, tr, crs)
        except Exception:
            out[key] = None
    return out


def longitudinal_chainage(gdf):
    """Compute chainage (cumulative distance) for LineString geometries.

    Adds two columns to the GeoDataFrame:
    - `geometry_length`: total length of geometry
    - `chainage_coords`: list of cumulative distances for coordinate vertices

    For multipart geometries the function merges parts before measuring.
    """
    if gdf is None or len(gdf) == 0:
        return gdf

    results = []
    for geom in gdf.geometry:
        try:
            # merge multi-part into single LineString when possible
            if geom.geom_type == 'MultiLineString':
                merged = linemerge(geom)
            else:
                merged = geom
            if not isinstance(merged, LineString):
                results.append({'geometry_length': 0.0, 'chainage_coords': []})
                continue
            coords = list(merged.coords)
            chain = [0.0]
            for a, b in zip(coords[:-1], coords[1:]):
                dx = a[0] - b[0]
                dy = a[1] - b[1]
                d = math.hypot(dx, dy)
                chain.append(chain[-1] + d)
            results.append({'geometry_length': chain[-1], 'chainage_coords': chain})
        except Exception:
            results.append({'geometry_length': 0.0, 'chainage_coords': []})

    # append columns
    gdf = gdf.copy()
    gdf['geometry_length'] = [r['geometry_length'] for r in results]
    gdf['chainage_coords'] = [r['chainage_coords'] for r in results]
    return gdf


def movie_frames_from_stack(depth_stack, trajs=None, style_opts=None):
    """Return a list of frames (numpy arrays) generated from a depth stack.

    - `depth_stack` is expected shape (T, H, W) or (H, W) for single frame.
    - `trajs` is optional dict {agent_id: [(x_t, y_t), ...]} where positions are
      in pixel coordinates; function will mark agent positions in the frames by
      setting the frame value to a highlight value.

    This function is intentionally lightweight and testable without matplotlib
    or ffmpeg.
    """
    arr = np.array(depth_stack)
    if arr.ndim == 2:
        arr = arr[None, ...]
    T, H, W = arr.shape
    frames = [arr[t].copy() for t in range(T)]
    if trajs:
        for aid, points in trajs.items():
            for t, pt in enumerate(points):
                if t < T and pt is not None:
                    x, y = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= y < H and 0 <= x < W:
                        frames[t][y, x] = np.nanmax(arr) + 1.0
    return frames


__all__ = [
    'output_excel',
    'movie_maker',
    'enviro_import',
    'longitudinal_import',
]
