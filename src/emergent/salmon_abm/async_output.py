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
import multiprocessing as mp
from multiprocessing import shared_memory

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

    def __init__(self, config: AsyncWriteConfig, *, h5obj: Any | None = None):
        self.config = config
        self._h5obj = h5obj
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
            h5_ctx = None
            h5 = self._h5obj
            if h5 is None:
                h5_ctx = h5py.File(cfg.h5_path, str(cfg.mode or "a"))
                h5 = h5_ctx

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
            if h5_ctx is not None:
                try:
                    h5_ctx.close()
                except Exception:
                    pass
        except BaseException as e:  # keep exception for propagation
            self._error = e


def _process_writer_main(cfg: AsyncWriteConfig, q, stop_evt) -> None:
    if h5py is None:
        return
    try:
        with h5py.File(cfg.h5_path, str(cfg.mode or "a")) as h5:
            last_flush_step = -1
            while True:
                if stop_evt.is_set() and q.empty():
                    break
                try:
                    item = q.get(timeout=0.25)
                except Exception:
                    continue
                if item is None:
                    break
                step, payload = item

                every = int(cfg.write_every_steps or 1)
                if every <= 0:
                    every = 1
                if int(step) % every != 0:
                    continue

                # Write payload (same semantics as ThreadHdfWriter).
                for key, arr in dict(payload).items():
                    k = str(key)
                    a = np.asarray(arr)
                    # Ensure groups exist for path-like keys.
                    try:
                        parts = k.split("/")
                        if len(parts) > 1:
                            grp = h5
                            for name in parts[:-1]:
                                if name:
                                    grp = grp.require_group(name)
                    except Exception:
                        pass
                    if k not in h5:
                        # create time-series dataset by default for agent_data/*
                        n_agents = int(cfg.n_agents)
                        dtype = a.dtype if a.dtype is not None else np.float32
                        compression = cfg.compression
                        compression_opts = cfg.compression_opts
                        if k.startswith("agent_data/"):
                            h5.create_dataset(
                                k,
                                shape=(n_agents, int(cfg.n_steps) if cfg.n_steps is not None else int(step) + 1),
                                maxshape=(n_agents, None) if cfg.n_steps is None else None,
                                dtype=dtype,
                                chunks=(min(n_agents, 1024), 1),
                                compression=compression,
                                compression_opts=compression_opts,
                            )
                        else:
                            h5.create_dataset(k, shape=(n_agents,), dtype=dtype, compression=compression, compression_opts=compression_opts)
                    ds = h5[k]
                    if ds.ndim == 1:
                        v = a.reshape((-1,))
                        ds[...] = v[: ds.shape[0]].astype(ds.dtype, copy=False)
                    elif ds.ndim == 2:
                        if cfg.n_steps is None and ds.shape[1] <= int(step):
                            ds.resize((ds.shape[0], int(step) + 1))
                        v = a.reshape((-1,))
                        ds[:, int(step)] = v[: ds.shape[0]].astype(ds.dtype, copy=False)

                flush_every = int(cfg.flush_every_steps or 0)
                if flush_every > 0 and int(step) != last_flush_step and (int(step) % flush_every == 0):
                    try:
                        h5.flush()
                    except Exception:
                        pass
                    last_flush_step = int(step)
            try:
                h5.flush()
            except Exception:
                pass
    except Exception:
        return


class ProcessHdfWriter:
    """Spawn-safe HDF5 writer process (single writer)."""

    def __init__(self, config: AsyncWriteConfig):
        self.config = config
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._queue = ctx.Queue(maxsize=max(1, int(config.queue_max or 64)))
        self._stop_evt = ctx.Event()
        self._proc = ctx.Process(target=_process_writer_main, args=(config, self._queue, self._stop_evt), daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._proc.start()

    def submit(self, step: int, payload: Mapping[str, Any], *, copy: bool = True) -> bool:
        if not self._started:
            self.start()
        # Normalize payload to numpy arrays; must be picklable.
        norm: dict[str, np.ndarray] = {}
        for k, v in dict(payload).items():
            if v is None:
                continue
            arr = np.asarray(v)
            if copy:
                arr = np.array(arr, copy=True)
            norm[str(k)] = arr

        policy = str(self.config.policy or "block").strip().lower()
        if policy not in ("block", "drop_oldest", "drop_newest"):
            policy = "block"
        try:
            if policy == "block":
                self._queue.put((int(step), norm), block=True, timeout=1.0)
                return True
            if policy == "drop_newest":
                self._queue.put((int(step), norm), block=False)
                return True
            # drop_oldest: try non-blocking; on full, drop one then retry
            try:
                self._queue.put((int(step), norm), block=False)
                return True
            except Exception:
                try:
                    _ = self._queue.get(block=False)
                except Exception:
                    pass
                try:
                    self._queue.put((int(step), norm), block=False)
                    return True
                except Exception:
                    return False
        except Exception:
            return False

    def close(self, timeout_s: float | None = None) -> bool:
        if not self._started:
            return True
        try:
            self._stop_evt.set()
        except Exception:
            pass
        try:
            self._queue.put(None, block=False)
        except Exception:
            pass
        try:
            self._proc.join(timeout=None if timeout_s is None else float(timeout_s))
        except Exception:
            pass
        return True


def _shm_ring_writer_main(cfg: AsyncWriteConfig, shm_meta: dict[str, tuple[str, str, tuple[int, int]]], steps_arr, sem_full, sem_empty, stop_evt) -> None:
    """Writer process that reads per-step payloads from a shared-memory ring buffer."""
    if h5py is None:
        return

    # Attach shared-memory segments and create numpy views.
    shms: dict[str, shared_memory.SharedMemory] = {}
    views: dict[str, np.ndarray] = {}
    try:
        for key, (name, dtype_str, shape) in dict(shm_meta).items():
            shm = shared_memory.SharedMemory(name=str(name), create=False)
            shms[key] = shm
            views[key] = np.ndarray(tuple(shape), dtype=np.dtype(dtype_str), buffer=shm.buf)
    except Exception:
        # Best-effort close
        for shm in shms.values():
            try:
                shm.close()
            except Exception:
                pass
        return

    try:
        with h5py.File(str(cfg.h5_path), str(cfg.mode or "a")) as h5:
            n_agents = int(cfg.n_agents)
            compression = cfg.compression
            compression_opts = cfg.compression_opts
            n_steps = cfg.n_steps
            ring_slots = int(next(iter(views.values())).shape[0]) if views else 0
            read_idx = 0
            last_flush_step = -1

            while True:
                try:
                    sem_full.acquire()
                except Exception:
                    break

                try:
                    step = int(steps_arr[read_idx])
                except Exception:
                    step = None

                # Sentinel ends the loop.
                if step is None or step < 0:
                    try:
                        sem_empty.release()
                    except Exception:
                        pass
                    break

                every = int(cfg.write_every_steps or 1)
                if every <= 0:
                    every = 1

                if int(step) % every == 0:
                    for k, arr in views.items():
                        try:
                            key = str(k)
                        except Exception:
                            continue
                        a = np.asarray(arr[read_idx, :]).reshape((-1,))

                        if key not in h5:
                            is_timeseries = str(key).startswith("agent_data/")
                            if (not is_timeseries):
                                h5.create_dataset(
                                    key,
                                    shape=(n_agents,),
                                    dtype=a.dtype,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                )
                            else:
                                if n_steps is None:
                                    init_t = int(step) + 1
                                    h5.create_dataset(
                                        key,
                                        shape=(n_agents, init_t),
                                        maxshape=(n_agents, None),
                                        dtype=a.dtype,
                                        chunks=(min(n_agents, 1024), 1),
                                        compression=compression,
                                        compression_opts=compression_opts,
                                    )
                                else:
                                    h5.create_dataset(
                                        key,
                                        shape=(n_agents, int(n_steps)),
                                        dtype=a.dtype,
                                        chunks=(min(n_agents, 1024), 1),
                                        compression=compression,
                                        compression_opts=compression_opts,
                                    )

                        ds = h5[key]
                        if ds.ndim == 1:
                            ds[...] = a[: ds.shape[0]].astype(ds.dtype, copy=False)
                        elif ds.ndim == 2:
                            if n_steps is None and ds.shape[1] <= int(step):
                                ds.resize((ds.shape[0], int(step) + 1))
                            ds[:, int(step)] = a[: ds.shape[0]].astype(ds.dtype, copy=False)

                    flush_every = int(cfg.flush_every_steps or 0)
                    if flush_every > 0 and int(step) != last_flush_step and (int(step) % flush_every == 0):
                        try:
                            h5.flush()
                        except Exception:
                            pass
                        last_flush_step = int(step)

                # Advance ring buffer.
                read_idx = (read_idx + 1) % max(1, int(ring_slots))
                try:
                    sem_empty.release()
                except Exception:
                    pass

            try:
                h5.flush()
            except Exception:
                pass
    finally:
        for shm in shms.values():
            try:
                shm.close()
            except Exception:
                pass


class ShmRingHdfWriter:
    """Spawn-safe shared-memory ring-buffer writer process (single writer).

    This reduces overhead vs `ProcessHdfWriter` by avoiding per-step pickling/IPC
    of large numpy arrays. The simulation process copies per-step values into a
    shared-memory ring buffer and signals the writer via semaphores.
    """

    def __init__(self, config: AsyncWriteConfig, *, keys: tuple[str, ...], ring_slots: int = 16, dtype: Any = np.float32):
        self.config = config
        self.keys = tuple(str(k) for k in keys)
        self.ring_slots = int(ring_slots)
        if self.ring_slots <= 0:
            self.ring_slots = 1
        self.dtype = np.dtype(dtype)

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._sem_empty = ctx.Semaphore(self.ring_slots)
        self._sem_full = ctx.Semaphore(0)
        self._stop_evt = ctx.Event()
        self._steps = ctx.Array("q", self.ring_slots, lock=False)  # int64 per slot

        self._shms: dict[str, shared_memory.SharedMemory] = {}
        self._views: dict[str, np.ndarray] = {}
        self._shm_meta: dict[str, tuple[str, str, tuple[int, int]]] = {}

        n_agents = int(config.n_agents)
        for key in self.keys:
            shm = shared_memory.SharedMemory(
                create=True,
                size=int(self.ring_slots) * int(n_agents) * int(self.dtype.itemsize),
            )
            self._shms[key] = shm
            view = np.ndarray((self.ring_slots, n_agents), dtype=self.dtype, buffer=shm.buf)
            self._views[key] = view
            self._shm_meta[key] = (shm.name, self.dtype.str, (self.ring_slots, n_agents))

        self._write_idx = 0
        self._proc = ctx.Process(
            target=_shm_ring_writer_main,
            args=(config, self._shm_meta, self._steps, self._sem_full, self._sem_empty, self._stop_evt),
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._proc.start()

    def submit(self, step: int, payload: Mapping[str, Any], *, copy: bool = False) -> bool:
        if not self._started:
            self.start()

        try:
            step_i = int(step)
        except Exception:
            return False

        try:
            if bool(getattr(self._stop_evt, "is_set", lambda: False)()):
                return False
        except Exception:
            pass

        # Backpressure: block until an empty slot is available.
        try:
            self._sem_empty.acquire()
        except Exception:
            return False

        idx = int(self._write_idx)
        try:
            pl = dict(payload)
        except Exception:
            pl = {}

        # Copy per-key arrays into shared memory slot.
        for key in self.keys:
            dst = self._views[key][idx, :]
            v = pl.get(key, None)
            if v is None:
                dst.fill(0.0)
                continue
            a = np.asarray(v, dtype=self.dtype).reshape((-1,))
            n = min(int(dst.shape[0]), int(a.shape[0]) if a.ndim == 1 else int(dst.shape[0]))
            if n > 0:
                dst[:n] = a[:n]
            if n < int(dst.shape[0]):
                dst[n:].fill(0.0)

        try:
            self._steps[idx] = int(step_i)
        except Exception:
            pass

        self._write_idx = (idx + 1) % int(self.ring_slots)
        try:
            self._sem_full.release()
        except Exception:
            return False
        return True

    def close(self, timeout_s: float | None = None) -> bool:
        if not self._started:
            # still need to unlink shared memory if we allocated it
            for shm in self._shms.values():
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            return True

        try:
            self._stop_evt.set()
        except Exception:
            pass

        # Send sentinel (-1) to ensure the writer wakes and exits.
        acquired = False
        try:
            if timeout_s is None:
                acquired = bool(self._sem_empty.acquire())
            else:
                acquired = bool(self._sem_empty.acquire(timeout=float(timeout_s)))
        except Exception:
            acquired = False
        if acquired:
            idx = int(self._write_idx)
            try:
                self._steps[idx] = -1
            except Exception:
                pass
            self._write_idx = (idx + 1) % int(self.ring_slots)
            try:
                self._sem_full.release()
            except Exception:
                pass
        try:
            self._proc.join(timeout=None if timeout_s is None else float(timeout_s))
        except Exception:
            pass

        try:
            if getattr(self._proc, "is_alive", lambda: False)():
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                return False
        except Exception:
            pass
        try:
            ec = getattr(self._proc, "exitcode", None)
            if ec not in (0, None):
                return False
        except Exception:
            pass

        # Cleanup shared memory segments (parent unlinks).
        for shm in self._shms.values():
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        return True
