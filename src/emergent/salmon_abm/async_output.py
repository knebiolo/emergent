"""Asynchronous output backends for Salmon ABM.

Phase 1 scope:
- Provide a no-op backend (NullWriter) for compute-only runs.
- Provide a thread-based HDF5 writer (ThreadHdfWriter) that owns all HDF5 writes.

Design notes:
- HDF5 is effectively single-writer; even in a thread backend, only the writer
  thread may touch the h5py.File / datasets.
- `submit()` defaults to copying payload arrays for safety; callers can disable
  copying when they manage buffer lifetimes externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
import threading
import time
from collections import deque

import numpy as np

try:  # optional dependency in some test contexts
    import h5py
except Exception:  # pragma: no cover
    h5py = None


@dataclass(frozen=True)
class AsyncWriteConfig:
    h5_path: str
    n_agents: int
    n_steps: Optional[int] = None
    mode: str = "a"
    queue_max: int = 64
    policy: str = "block"  # block | drop_oldest | drop_newest
    write_every_steps: int = 1
    flush_every_steps: int = 0
    compression: Optional[str] = None
    compression_opts: Any = None


class NullWriter:
    def start(self) -> None:
        return None

    def submit(self, step: int, payload: Mapping[str, Any], *, copy: bool = False) -> bool:
        return True

    def close(self, timeout_s: float | None = None) -> bool:
        return True


class ThreadHdfWriter:
    """Threaded HDF5 writer with bounded queue and backpressure policy."""

    def __init__(self, config: AsyncWriteConfig):
        self.config = config
        self._cond = threading.Condition()
        self._queue: deque[tuple[int, dict[str, np.ndarray]]] = deque()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        t = threading.Thread(target=self._run, name="ThreadHdfWriter", daemon=True)
        self._thread = t
        t.start()

    def submit(self, step: int, payload: Mapping[str, Any], *, copy: bool = True) -> bool:
        if not self._started:
            self.start()
        if self._error is not None:
            raise RuntimeError("Async writer previously failed") from self._error

        try:
            step_i = int(step)
        except Exception:
            step_i = step  # type: ignore[assignment]

        # Normalize payload to numpy arrays (optionally copying for safety).
        norm: dict[str, np.ndarray] = {}
        for k, v in dict(payload).items():
            if v is None:
                continue
            arr = np.asarray(v)
            if copy:
                arr = np.array(arr, copy=True)
            norm[str(k)] = arr

        with self._cond:
            maxn = int(self.config.queue_max)
            if maxn <= 0:
                maxn = 1
            policy = str(self.config.policy or "block").strip().lower()
            if policy not in ("block", "drop_oldest", "drop_newest"):
                policy = "block"

            while (not self._stop) and len(self._queue) >= maxn and policy == "block":
                self._cond.wait(timeout=0.25)

            if self._stop:
                return False

            if len(self._queue) >= maxn:
                if policy == "drop_oldest":
                    try:
                        self._queue.popleft()
                    except Exception:
                        pass
                elif policy == "drop_newest":
                    return False

            self._queue.append((step_i, norm))
            self._cond.notify_all()
            return True

    def close(self, timeout_s: float | None = None) -> bool:
        if not self._started:
            return True
        with self._cond:
            self._stop = True
            self._cond.notify_all()

        thr = self._thread
        if thr is not None:
            thr.join(timeout=None if timeout_s is None else float(timeout_s))
        if self._error is not None:
            raise RuntimeError("Async writer failed") from self._error
        return True

    def _ensure_dataset(self, h5, key: str, *, step: int, values: np.ndarray):
        if h5py is None:
            raise RuntimeError("h5py not available")

        if key in h5:
            return h5[key]

        # Create dataset lazily.
        n_agents = int(self.config.n_agents)
        vals = np.asarray(values)
        dtype = vals.dtype if vals.dtype is not None else np.float32

        compression = self.config.compression
        compression_opts = self.config.compression_opts

        # Heuristic: `agent_data/*` is treated as time-indexed (N, T) even when
        # values submitted per step are 1-D arrays.
        is_timeseries = str(key).startswith("agent_data/")
        if (not is_timeseries) and vals.ndim == 1 and vals.shape[0] == n_agents:
            return h5.create_dataset(
                key,
                shape=(n_agents,),
                dtype=dtype,
                compression=compression,
                compression_opts=compression_opts,
            )

        # Default to a time-indexed dataset for agent series (N, T).
        n_steps = self.config.n_steps
        if n_steps is None:
            # resizable along time axis
            init_t = int(step) + 1
            return h5.create_dataset(
                key,
                shape=(n_agents, init_t),
                maxshape=(n_agents, None),
                dtype=dtype,
                chunks=(min(n_agents, 1024), 1),
                compression=compression,
                compression_opts=compression_opts,
            )
        return h5.create_dataset(
            key,
            shape=(n_agents, int(n_steps)),
            dtype=dtype,
            chunks=(min(n_agents, 1024), 1),
            compression=compression,
            compression_opts=compression_opts,
        )

    def _write_payload(self, h5, step: int, payload: dict[str, np.ndarray]) -> None:
        for key, arr in payload.items():
            ds = self._ensure_dataset(h5, key, step=step, values=arr)
            a = np.asarray(arr)
            if ds.ndim == 1:
                v = a.reshape((-1,))
                try:
                    ds[...] = v[: ds.shape[0]]
                except Exception:
                    ds[...] = np.asarray(v[: ds.shape[0]], dtype=ds.dtype)
                continue

            if ds.ndim == 2:
                if self.config.n_steps is None and ds.shape[1] <= int(step):
                    ds.resize((ds.shape[0], int(step) + 1))
                v = a.reshape((-1,))
                ds[:, int(step)] = v[: ds.shape[0]]
                continue

            # Unknown dataset shape: skip
            continue

    def _run(self) -> None:
        if h5py is None:
            self._error = RuntimeError("h5py not available")
            return
        cfg = self.config
        try:
            with h5py.File(cfg.h5_path, str(cfg.mode or "a")) as h5:
                last_flush_step = -1
                while True:
                    item = None
                    with self._cond:
                        while (not self._stop) and (len(self._queue) == 0):
                            self._cond.wait(timeout=0.25)
                        if len(self._queue) > 0:
                            item = self._queue.popleft()
                            self._cond.notify_all()
                        elif self._stop:
                            break
                    if item is None:
                        continue
                    step, payload = item

                    every = int(cfg.write_every_steps or 1)
                    if every <= 0:
                        every = 1
                    if int(step) % every != 0:
                        continue

                    self._write_payload(h5, int(step), payload)

                    flush_every = int(cfg.flush_every_steps or 0)
                    if flush_every > 0 and int(step) != last_flush_step and (int(step) % flush_every == 0):
                        try:
                            h5.flush()
                        except Exception:
                            pass
                        last_flush_step = int(step)

                # Final flush
                try:
                    h5.flush()
                except Exception:
                    pass
        except BaseException as e:  # keep exception for propagation
            self._error = e
