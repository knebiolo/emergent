"""Small helpers to centralize HDF5 access for salmon_abm.

This module provides thin wrappers around h5py.File-style objects
but also accepts dict-like mocks used in unit tests.

Functions are intentionally small and conservative to make gradual
refactoring low-risk.
"""
from contextlib import contextmanager
from typing import Any, Optional, Iterable
import os
import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - h5py should be available in normal envs
    h5py = None


def _is_h5py_file(obj: Any) -> bool:
    return h5py is not None and isinstance(obj, h5py.File)


@contextmanager
def open_db(path: str, mode: str = "r"):
    """Context manager that opens an HDF5 file and yields the file object.

    If h5py is not available this will raise. Tests can pass in dict-like
    objects directly and avoid using this helper.
    """
    if h5py is None:
        raise RuntimeError("h5py not available in this environment")
    f = h5py.File(path, mode)
    try:
        yield f
    finally:
        try:
            f.close()
        except Exception:
            pass


def read_dataset(hdf5_obj: Any, key: str, default: Optional[Any] = None):
    """Read a dataset or key from an HDF5 file or dict-like object.

    Returns the dataset value (copied for h5py datasets) or `default` if missing.
    """
    try:
        if hdf5_obj is None:
            return default
        if hasattr(hdf5_obj, "__getitem__"):
            val = hdf5_obj[key]
            # if this is an h5py dataset, return a numpy copy
            if hasattr(val, "[:]"):
                try:
                    return val[:]
                except Exception:
                    return np.array(val)
            return val
    except Exception:
        return default


def write_dataset(hdf5_obj: Any, key: str, data: Any, dtype: Optional[str] = None):
    """Write `data` to `key` in the HDF5 file-like object.

    If the key exists it will be overwritten if possible. For dict-like
    objects this sets the key directly.
    """
    if hdf5_obj is None:
        return False
    # dict-like
    if isinstance(hdf5_obj, dict):
        hdf5_obj[key] = np.array(data)
        return True

    # h5py
    try:
        if key in hdf5_obj:
            del hdf5_obj[key]
        if dtype:
            hdf5_obj.create_dataset(key, data=np.array(data), dtype=dtype)
        else:
            hdf5_obj.create_dataset(key, data=np.array(data))
        return True
    except Exception:
        return False


def ensure_group(hdf5_obj: Any, group: str):
    """Ensure a group exists in the hdf5 file or dict-like; return the group.

    For dict-like objects, groups are represented as nested dicts.
    """
    if isinstance(hdf5_obj, dict):
        if group not in hdf5_obj:
            hdf5_obj[group] = {}
        return hdf5_obj[group]

    try:
        if group in hdf5_obj:
            return hdf5_obj[group]
        return hdf5_obj.create_group(group)
    except Exception:
        return None


def create_environment_placeholders(hdf5_obj: Any):
    """Create small placeholder datasets used by other modules.

    This function creates shallow minimal datasets so downstream code
    that expects keys like 'environment/depth' or 'memory/0' can proceed.
    """
    # depth and coordinate placeholders
    write_dataset(hdf5_obj, "environment/depth", np.zeros((1, 1), dtype=np.float32))
    write_dataset(hdf5_obj, "environment/x_coords", np.array([0.0], dtype=np.float32))
    write_dataset(hdf5_obj, "environment/y_coords", np.array([0.0], dtype=np.float32))

    # simple memory placeholders
    # some tests expect memory/0 and memory/1
    write_dataset(hdf5_obj, "memory/0", np.zeros((1, 1), dtype=np.float32))
    write_dataset(hdf5_obj, "memory/1", np.zeros((1, 1), dtype=np.float32))
    return True


def sample_environment(hdf5_obj: Any, x_idx: int, y_idx: int, band: str = "depth"):
    """Return the value at (x_idx, y_idx) from environment/<band>.

    Returns np.nan if out-of-bounds or missing.
    """
    ds = read_dataset(hdf5_obj, f"environment/{band}", default=None)
    try:
        if ds is None:
            return np.nan
        return ds[y_idx, x_idx]
    except Exception:
        return np.nan


def get_hdf5_obj(simulation_obj: Any):
    """Return an HDF5-like object from a simulation object.

    Prefers `simulation_obj.hdf5` if present, otherwise `simulation_obj.db`.
    Returns None if neither exists.
    """
    if simulation_obj is None:
        return None
    if hasattr(simulation_obj, "hdf5"):
        return getattr(simulation_obj, "hdf5")
    if hasattr(simulation_obj, "db"):
        return getattr(simulation_obj, "db")
    return None
