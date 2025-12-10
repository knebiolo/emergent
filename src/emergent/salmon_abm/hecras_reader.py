"""Minimal HECRAS HDF reader utilities.

This module provides a small, dependency-light API to extract cell
center coordinates and a numeric value (depth/ele) from HECRAS HDF5
files produced by the project. The intent is to keep this focused and
easy to test; it can be expanded later with more sophisticated
filtering, CRS handling, or raster fallbacks if needed.

Functions
- load_hecras_cells(path, value_dataset_name='Depth') -> (pts, vals):
    Read cell center XY coordinates and a 1D array of values.
"""
from typing import Tuple
import numpy as np
import h5py


def _safe_get(h5, key: str):
    if key in h5:
        return h5[key]
    # try common alternatives
    for k in h5.keys():
        if k.lower().endswith(key.lower()):
            return h5[k]
    return None


def load_hecras_cells(path: str, value_dataset_name: str = 'Depth') -> Tuple[np.ndarray, np.ndarray]:
    """Load HECRAS cell centers and a value array from an HDF5 file.

    Returns:
      pts: (N, 2) float array of X, Y cell centers
      vals: (N,) float array matching pts

    The function is defensive: it raises IOError on missing files and
    ValueError for missing expected datasets.
    """
    try:
        f = h5py.File(path, 'r')
    except Exception as e:
        raise IOError(f"Unable to open HDF5 file '{path}': {e}")

    with f:
        # common HECRAS layouts vary; try several likely dataset names
        # cell centers are often stored as 'CellCenters' or similar
        centers_ds = _safe_get(f, 'CellCenters')
        if centers_ds is None:
            # try combined XY arrays
            xs = _safe_get(f, 'X')
            ys = _safe_get(f, 'Y')
            if xs is not None and ys is not None:
                pts = np.vstack([np.array(xs).ravel(), np.array(ys).ravel()]).T
            else:
                raise ValueError("Unable to find cell center datasets in HDF5")
        else:
            pts = np.array(centers_ds)
            if pts.ndim == 1 and pts.size % 2 == 0:
                pts = pts.reshape(-1, 2)
            elif pts.ndim == 2 and pts.shape[1] >= 2:
                pts = pts[:, :2]

        # load the requested value dataset
        vals_ds = _safe_get(f, value_dataset_name)
        if vals_ds is None:
            # try common alternatives
            for candidate in ['Depth', 'WaterDepth', 'Elevation', 'ELE']:
                vals_ds = _safe_get(f, candidate)
                if vals_ds is not None:
                    break
        if vals_ds is None:
            # fallback: attempt to find a numeric 1D dataset with matching length
            for k in f.keys():
                ds = f[k]
                try:
                    arr = np.array(ds)
                except Exception:
                    continue
                if arr.ndim == 1 and arr.size == pts.shape[0]:
                    vals_ds = ds
                    break

        if vals_ds is None:
            raise ValueError(f"Unable to locate a value dataset matching '{value_dataset_name}'")

        vals = np.array(vals_ds).ravel()

        if vals.shape[0] != pts.shape[0]:
            # try to broadcast or trim if possible
            if vals.size == 1:
                vals = np.repeat(vals.item(), pts.shape[0])
            else:
                raise ValueError("Value array length does not match number of points")

        return pts.astype(float), vals.astype(float)


__all__ = ["load_hecras_cells"]
