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


def create_agent_timeseries(hdf5_obj: Any, sim: Any, nsteps: int):
    """Create agent time-series and scalar datasets used by the simulation runner.

    This helper centralizes dataset names and shapes so callers (scripts/tools)
    remain small. It will not overwrite existing datasets if present.
    """
    if hdf5_obj is None or sim is None:
        return False

    n_agents = getattr(sim, 'X', None)
    if n_agents is None:
        return False
    try:
        n_agents = int(np.array(getattr(sim, 'X')).shape[0])
    except Exception:
        return False

    shape = (n_agents, int(nsteps))

    def _ensure(key, default_arr):
        if key in hdf5_obj:
            return
        try:
            # write_dataset will handle dict-like or h5py objects
            write_dataset(hdf5_obj, key, default_arr)
        except Exception:
            pass

    # per-timestep 2D slots (num_agents x nsteps)
    _ensure('agent_data/X', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/Y', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/prev_X', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/prev_Y', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/x_vel', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/y_vel', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/Hz', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/heading', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/swim_behav', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/is_stuck', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/thrust', np.zeros(shape, dtype=np.float32))
    _ensure('agent_data/drag', np.zeros(shape, dtype=np.float32))

    # 1-D per-agent scalars (parity with sockeye.py)
    try:
        if hasattr(sim, 'length') and 'agent_data/length' not in hdf5_obj:
            write_dataset(hdf5_obj, 'agent_data/length', np.array(getattr(sim, 'length')))
        if hasattr(sim, 'weight') and 'agent_data/weight' not in hdf5_obj:
            write_dataset(hdf5_obj, 'agent_data/weight', np.array(getattr(sim, 'weight')))
        if hasattr(sim, 'ucrit') and 'agent_data/ucrit' not in hdf5_obj:
            write_dataset(hdf5_obj, 'agent_data/ucrit', np.array(getattr(sim, 'ucrit')))
        if hasattr(sim, 'too_shallow') and 'agent_data/too_shallow' not in hdf5_obj:
            write_dataset(hdf5_obj, 'agent_data/too_shallow', np.array(getattr(sim, 'too_shallow')))
        if hasattr(sim, 'opt_wat_depth') and 'agent_data/opt_wat_depth' not in hdf5_obj:
            write_dataset(hdf5_obj, 'agent_data/opt_wat_depth', np.array(getattr(sim, 'opt_wat_depth')))
    except Exception:
        pass

    return True


def write_agent_timestep(hdf5_obj: Any, sim: Any, col: int):
    """Write the current sim per-agent state into agent_data/* at column `col`.

    This attempts to write slices into existing datasets; missing keys are
    silently skipped to keep callers simple.
    """
    if hdf5_obj is None or sim is None:
        return False

    keys_and_getters = [
        ('agent_data/X', lambda s: getattr(s, 'X', None)),
        ('agent_data/Y', lambda s: getattr(s, 'Y', None)),
        ('agent_data/x_vel', lambda s: getattr(s, 'x_vel', None)),
        ('agent_data/y_vel', lambda s: getattr(s, 'y_vel', None)),
        ('agent_data/Hz', lambda s: getattr(s, 'Hz', None)),
        ('agent_data/heading', lambda s: getattr(s, 'heading', None)),
        ('agent_data/swim_behav', lambda s: getattr(s, 'swim_behav', None)),
        ('agent_data/is_stuck', lambda s: getattr(s, 'is_stuck', None)),
        ('agent_data/thrust', lambda s: (np.linalg.norm(getattr(s, 'thrust', np.zeros((len(getattr(s, "X")),2))), axis=1) if hasattr(s, 'thrust') else None)),
        ('agent_data/drag', lambda s: (np.linalg.norm(getattr(s, 'drag', np.zeros((len(getattr(s, "X")),2))), axis=1) if hasattr(s, 'drag') else None)),
    ]

    for key, getter in keys_and_getters:
        try:
            if key not in hdf5_obj:
                continue
            val = getter(sim)
            if val is None:
                continue
            # assign column for h5py datasets and dict-like arrays
            try:
                hdf5_obj[key][:, col] = np.array(val)
            except Exception:
                # fallback: overwrite whole dataset (less efficient)
                try:
                    write_dataset(hdf5_obj, key, np.array(val))
                except Exception:
                    pass
        except Exception:
            pass

    return True
