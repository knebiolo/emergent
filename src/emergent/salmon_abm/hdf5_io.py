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
import logging

try:
    import h5py
except Exception:  # pragma: no cover - h5py should be available in normal envs
    h5py = None

logger = logging.getLogger(__name__)


def _is_h5py_file(obj: Any) -> bool:
    return h5py is not None and isinstance(obj, h5py.File)


def _key_exists(hdf5_obj: Any, key: str) -> bool:
    """Return True if `key` exists in `hdf5_obj`.

    Works for h5py objects and dict-like mocks. Never raises.
    """
    if hdf5_obj is None:
        return False
    try:
        return key in hdf5_obj
    except Exception:
        # Some dict-like mocks may not implement `in` reliably.
        try:
            _ = hdf5_obj[key]
            return True
        except Exception:
            return False


def _h5_is_dataset(obj: Any) -> bool:
    """Return True if `obj` is an h5py.Dataset (when h5py is available)."""
    if h5py is None:
        return False
    try:
        return isinstance(obj, h5py.Dataset)
    except Exception:
        return False


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
        except (OSError, ValueError):
            logger.debug("Failed closing HDF5 file: %s", path, exc_info=True)


def read_dataset(hdf5_obj: Any, key: str, default: Optional[Any] = None):
    """Read a dataset or key from an HDF5 file or dict-like object.

    Returns the dataset value (copied for h5py datasets) or `default` if missing.
    """
    if hdf5_obj is None or not hasattr(hdf5_obj, "__getitem__"):
        return default

    try:
        val = hdf5_obj[key]
    except Exception:
        return default

    # h5py dataset: always materialize a numpy array for downstream code
    if _h5_is_dataset(val):
        try:
            return val[...]
        except Exception:
            try:
                return np.array(val)
            except Exception:
                return default

    return val


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
        arr = np.array(data)
        if _key_exists(hdf5_obj, key):
            try:
                existing = hdf5_obj[key]
            except Exception:
                existing = None

            # Prefer in-place overwrite for existing datasets when shape/dtype match.
            if _h5_is_dataset(existing):
                try:
                    if existing.shape == arr.shape:
                        if dtype is None or existing.dtype == np.dtype(dtype):
                            existing[...] = arr.astype(existing.dtype, copy=False)
                            return True
                except (OSError, ValueError, TypeError):
                    # Fall back to delete/recreate below.
                    logger.debug("In-place HDF5 dataset overwrite failed for key=%s", key, exc_info=True)

            try:
                del hdf5_obj[key]
            except (KeyError, OSError, ValueError, TypeError):
                # If we can't delete, fall through to create (may raise)
                logger.debug("Failed deleting existing HDF5 key=%s before recreate", key, exc_info=True)

        if dtype:
            hdf5_obj.create_dataset(key, data=arr, dtype=dtype)
        else:
            hdf5_obj.create_dataset(key, data=arr)
        return True
    except Exception:
        return False


def write_timeseries_step(hdf5_obj: Any, key: str, step: int, values: Any) -> bool:
    """Write a single timestep column into a 2D (agents x timesteps) dataset.

    This avoids materializing and rewriting the full 2D array each timestep.

    - For h5py datasets: assigns `ds[:, step] = values` in-place.
    - For dict-like stores: assigns into the existing numpy array when present.

    Returns False if the dataset is missing or cannot be written.
    """
    if hdf5_obj is None:
        return False
    step_i = int(step)
    vals = np.asarray(values)

    if isinstance(hdf5_obj, dict):
        arr = hdf5_obj.get(key)
        if arr is None:
            return False
        try:
            arr = np.asarray(arr)
            if arr.ndim != 2 or step_i < 0 or step_i >= arr.shape[1]:
                return False
            arr[:, step_i] = vals.reshape((-1,))
            hdf5_obj[key] = arr
            return True
        except Exception:
            return False

    # h5py-backed: write column slice without copying full dataset
    try:
        ds = hdf5_obj[key]
    except Exception:
        return False
    if not _h5_is_dataset(ds):
        return False
    try:
        if ds.ndim != 2 or step_i < 0 or step_i >= ds.shape[1]:
            return False
        ds[:, step_i] = vals.reshape((-1,)).astype(ds.dtype, copy=False)
        return True
    except Exception:
        return False


def _ensure_h5_groups_for_key(hdf5_obj: Any, key: str) -> None:
    """Ensure any intermediate groups in a path-like key exist (h5py only)."""
    if h5py is None or hdf5_obj is None:
        return
    try:
        if not hasattr(hdf5_obj, "require_group"):
            return
    except Exception:
        return
    try:
        parts = str(key).split("/")
    except Exception:
        return
    if len(parts) <= 1:
        return
    grp = hdf5_obj
    for name in parts[:-1]:
        if not name:
            continue
        try:
            grp = grp.require_group(name)
        except Exception:
            return


def ensure_timeseries_dataset(
    hdf5_obj: Any,
    key: str,
    *,
    n_agents: int,
    n_steps: int,
    dtype: Any = np.float32,
    chunks: tuple[int, int] | None = None,
    fillvalue: Any = 0.0,
    compression: Optional[str] = None,
    compression_opts: Any = None,
) -> bool:
    """Ensure a 2D (agents x timesteps) dataset exists without writing full data."""
    if hdf5_obj is None:
        return False
    try:
        key = str(key)
    except Exception:
        raise
    if _key_exists(hdf5_obj, key):
        return True

    na = max(0, int(n_agents))
    nt = max(0, int(n_steps))
    if na <= 0 or nt <= 0:
        return False

    if isinstance(hdf5_obj, dict):
        hdf5_obj[key] = np.zeros((na, nt), dtype=dtype)
        return True

    if h5py is None:
        return False

    try:
        _ensure_h5_groups_for_key(hdf5_obj, key)
        if chunks is None:
            chunks = (min(na, 1024), 1)
        hdf5_obj.create_dataset(
            key,
            shape=(na, nt),
            dtype=dtype,
            chunks=chunks,
            fillvalue=fillvalue,
            compression=compression,
            compression_opts=compression_opts,
        )
        return True
    except Exception:
        return False


def ensure_vector_dataset(
    hdf5_obj: Any,
    key: str,
    *,
    n_agents: int,
    dtype: Any = np.float32,
    fillvalue: Any = 0.0,
    compression: Optional[str] = None,
    compression_opts: Any = None,
) -> bool:
    """Ensure a 1D (agents,) dataset exists without writing full data."""
    if hdf5_obj is None:
        return False
    try:
        key = str(key)
    except Exception:
        raise
    if _key_exists(hdf5_obj, key):
        return True

    na = max(0, int(n_agents))
    if na <= 0:
        return False

    if isinstance(hdf5_obj, dict):
        hdf5_obj[key] = np.full((na,), fillvalue, dtype=dtype)
        return True

    if h5py is None:
        return False

    try:
        _ensure_h5_groups_for_key(hdf5_obj, key)
        hdf5_obj.create_dataset(
            key,
            shape=(na,),
            dtype=dtype,
            fillvalue=fillvalue,
            compression=compression,
            compression_opts=compression_opts,
        )
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
        if _key_exists(hdf5_obj, group):
            return hdf5_obj[group]
        return hdf5_obj.create_group(group)
    except Exception:
        return None


def create_environment_placeholders(hdf5_obj: Any):
    """Create small placeholder datasets used by other modules.

    This function creates shallow minimal datasets so downstream code
    that expects keys like 'environment/depth' or 'memory/0' can proceed.
    """
    def _ensure(key: str, arr: np.ndarray) -> None:
        if _key_exists(hdf5_obj, key):
            return
        write_dataset(hdf5_obj, key, arr)

    # depth and coordinate placeholders (only when missing)
    _ensure("environment/depth", np.zeros((1, 1), dtype=np.float32))
    _ensure("environment/x_coords", np.array([0.0], dtype=np.float32))
    _ensure("environment/y_coords", np.array([0.0], dtype=np.float32))

    # simple memory placeholders (only when missing)
    _ensure("memory/0", np.zeros((1, 1), dtype=np.float32))
    _ensure("memory/1", np.zeros((1, 1), dtype=np.float32))
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

    nsteps_i = int(nsteps)
    shape = (n_agents, nsteps_i)

    # per-timestep 2D slots (num_agents x nsteps) created chunked for column writes
    for key in (
        'agent_data/X',
        'agent_data/Y',
        'agent_data/prev_X',
        'agent_data/prev_Y',
        'agent_data/x_vel',
        'agent_data/y_vel',
        'agent_data/Hz',
        'agent_data/heading',
        'agent_data/swim_behav',
        'agent_data/is_stuck',
        'agent_data/thrust',
        'agent_data/drag',
    ):
        ensure_timeseries_dataset(hdf5_obj, key, n_agents=n_agents, n_steps=nsteps_i, dtype=np.float32)

    # 1-D per-agent scalars (parity with sockeye.py)
    if hasattr(sim, 'length') and not _key_exists(hdf5_obj, 'agent_data/length'):
        write_dataset(hdf5_obj, 'agent_data/length', np.array(getattr(sim, 'length')))
    if hasattr(sim, 'weight') and not _key_exists(hdf5_obj, 'agent_data/weight'):
        write_dataset(hdf5_obj, 'agent_data/weight', np.array(getattr(sim, 'weight')))
    if hasattr(sim, 'ucrit') and not _key_exists(hdf5_obj, 'agent_data/ucrit'):
        write_dataset(hdf5_obj, 'agent_data/ucrit', np.array(getattr(sim, 'ucrit')))
    if hasattr(sim, 'too_shallow') and not _key_exists(hdf5_obj, 'agent_data/too_shallow'):
        write_dataset(hdf5_obj, 'agent_data/too_shallow', np.array(getattr(sim, 'too_shallow')))
    if hasattr(sim, 'opt_wat_depth') and not _key_exists(hdf5_obj, 'agent_data/opt_wat_depth'):
        write_dataset(hdf5_obj, 'agent_data/opt_wat_depth', np.array(getattr(sim, 'opt_wat_depth')))

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
        if not _key_exists(hdf5_obj, key):
            continue
        val = getter(sim)
        if val is None:
            continue

        arr = np.array(val)
        try:
            hdf5_obj[key][:, col] = arr
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
            logger.debug("Column write failed for key=%s at col=%s; overwriting dataset", key, col, exc_info=True)
            ok = write_dataset(hdf5_obj, key, arr)
            if not ok:
                raise RuntimeError(f"Failed writing dataset for key={key}") from e

    return True
