import os
import h5py
import numpy as np


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _h5_bytes(s: str):
    try:
        return np.string_(s)
    except Exception:
        return np.bytes_(s.encode("utf-8", errors="replace"))


def _safe_del(group, key: str) -> None:
    if key in group:
        del group[key]


def _write_value(group, key: str, value) -> None:
    """Write a value to an h5py group, best-effort with string fallback."""
    try:
        arr = np.asarray(value)
        _safe_del(group, key)
        if arr.ndim == 0:
            group.create_dataset(key, data=arr)
        else:
            group.create_dataset(key, data=arr, compression="gzip")
    except Exception:
        try:
            _safe_del(group, key)
            group.create_dataset(key, data=_h5_bytes(str(value)))
        except Exception:
            pass


def _flush_file(file) -> None:
    try:
        file.flush()
    except Exception:
        return
    try:
        os.fsync(file.id.fileno())
    except Exception:
        pass


class HDF5DiagnosticsWriter:
    def __init__(self, path):
        self.path = path
        self.file = None

    def open(self, mode='a'):
        _ensure_parent_dir(self.path)
        # use swmr disabled for simplicity; opens in append/create mode
        self.file = h5py.File(self.path, mode)

    def close(self):
        if self.file is None:
            return
        try:
            self.file.close()
        except Exception:
            pass
        finally:
            self.file = None

    def write_step(self, step, payload: dict):
        if self.file is None:
            raise RuntimeError('Diagnostics file not open')
        grp_name = f'steps/{int(step)}'
        if grp_name in self.file:
            grp = self.file[grp_name]
        else:
            grp = self.file.create_group(grp_name)
        # write datasets for arrays in payload; overwrite if exists
        for k, v in payload.items():
            _write_value(grp, str(k), v)
        # flush to disk to minimize window where data is missing
        _flush_file(self.file)
