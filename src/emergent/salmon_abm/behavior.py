"""Behavior helpers extracted from sockeye.py.

This includes perception and social cue calculations.
"""
import os
import numpy as np
import pandas as pd
import logging
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import UnivariateSpline
import h5py
import time
import sys
import threading
import queue
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False
    

from emergent.salmon_abm.utils import geo_to_pixel, pixel_to_geo, _unpack_affine, standardize_shape, calculate_front_masks, determine_slices_from_vectors, determine_slices_from_headings
from emergent.salmon_abm import hdf5_io

# Optional Numba JIT: use if available to accelerate inner loops
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _repulsive_core(agent_x, agent_y, world_x, world_y, multiplier, weight):
        total_x = 0.0
        total_y = 0.0
        n = world_x.shape[0]
        for i in prange(n):
            dx = agent_x - world_x[i]
            dy = agent_y - world_y[i]
            mag = (dx * dx + dy * dy) ** 0.5
            if mag == 0.0:
                mag = 1e-6
            ux = dx / mag
            uy = dy / mag
            m = multiplier[i]
            fx = ((weight * ux) / mag) * m
            fy = ((weight * uy) / mag) * m
            total_x += fx
            total_y += fy
        return total_x, total_y

    def _repulsive_core_safe(agent_x, agent_y, world_x, world_y, multiplier, weight):
        # Ensure inputs are contiguous float64 1-D arrays for Numba
        wx = np.ascontiguousarray(world_x, dtype=np.float64)
        wy = np.ascontiguousarray(world_y, dtype=np.float64)
        mult = np.ascontiguousarray(multiplier, dtype=np.float64)
        return _repulsive_core(agent_x, agent_y, wx, wy, mult, float(weight))

    @njit(parallel=True, fastmath=True)
    def _repulsive_batched_core(agent_xs, agent_ys, mmap_flat, mmap_offsets, rows_min, cols_min, nr_list, nc_list, affine, weight, t, out_x, out_y):
        # affine: 6-element array (a,b,c,d,e,f)
        a = affine[0]
        b = affine[1]
        c = affine[2]
        d = affine[3]
        e = affine[4]
        f = affine[5]
        n_agents = agent_xs.shape[0]
        for aidx in prange(n_agents):
            ax = agent_xs[aidx]
            ay = agent_ys[aidx]
            start = mmap_offsets[aidx]
            nr = nr_list[aidx]
            nc = nc_list[aidx]
            tx = 0.0
            ty = 0.0
            if nr <= 0 or nc <= 0:
                out_x[aidx] = 0.0
                out_y[aidx] = 0.0
                continue
            # iterate rows and cols
            for ri in range(nr):
                for ci in range(nc):
                    idx = start + ri * nc + ci
                    val = mmap_flat[idx]
                    t_since = t - val
                    m = 0.0
                    if t_since > 10.0 and t_since < 7200.0:
                        m = 1.0 - (t_since - 5.0) / 7195.0
                    row = rows_min[aidx] + ri
                    col = cols_min[aidx] + ci
                    wx = a * col + b * row + c
                    wy = d * col + e * row + f
                    dx = ax - wx
                    dy = ay - wy
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag == 0.0:
                        mag = 1e-6
                    ux = dx / mag
                    uy = dy / mag
                    fx = ((weight * ux) / mag) * m
                    fy = ((weight * uy) / mag) * m
                    tx += fx
                    ty += fy
            out_x[aidx] = tx
            out_y[aidx] = ty

    def _repulsive_batched_core_safe(agent_xs, agent_ys, mmap_flat, mmap_offsets, rows_min, cols_min, nr_list, nc_list, affine, weight, t):
        mf = np.ascontiguousarray(mmap_flat, dtype=np.float64)
        mo = np.ascontiguousarray(mmap_offsets, dtype=np.int64)
        rm = np.ascontiguousarray(rows_min, dtype=np.int64)
        cm = np.ascontiguousarray(cols_min, dtype=np.int64)
        nr = np.ascontiguousarray(nr_list, dtype=np.int64)
        nc = np.ascontiguousarray(nc_list, dtype=np.int64)
        axs = np.ascontiguousarray(agent_xs, dtype=np.float64)
        ays = np.ascontiguousarray(agent_ys, dtype=np.float64)
        aff = np.ascontiguousarray(affine, dtype=np.float64)
        out_x = np.empty(axs.shape[0], dtype=np.float64)
        out_y = np.empty(axs.shape[0], dtype=np.float64)
        _repulsive_batched_core(axs, ays, mf, mo, rm, cm, nr, nc, aff, float(weight), float(t), out_x, out_y)
        return out_x, out_y

    @njit(parallel=True, fastmath=True)
    def _cohesion_from_neighbors_offsets(
        offsets_i64,
        neighbors_i32,
        X_f64,
        Y_f64,
        x_vel_f64,
        y_vel_f64,
        consider_front_only_i8,
        out_unit_x_f64,
        out_unit_y_f64,
    ):
        n = offsets_i64.shape[0] - 1
        for i in prange(n):
            start = offsets_i64[i]
            end = offsets_i64[i + 1]
            if start >= end:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
                continue
            ax = X_f64[i]
            ay = Y_f64[i]
            vx_i = x_vel_f64[i]
            vy_i = y_vel_f64[i]
            sx = 0.0
            sy = 0.0
            cnt = 0
            for j in range(start, end):
                nb = neighbors_i32[j]
                if nb < 0:
                    continue
                if nb == i:
                    continue
                if consider_front_only_i8 != 0:
                    dx = X_f64[nb] - ax
                    dy = Y_f64[nb] - ay
                    if (dx * vx_i + dy * vy_i) <= 0.0:
                        continue
                sx += X_f64[nb]
                sy += Y_f64[nb]
                cnt += 1
            if cnt <= 0:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
                continue
            cx = sx / cnt
            cy = sy / cnt
            dx = cx - ax
            dy = cy - ay
            mag = (dx * dx + dy * dy) ** 0.5
            if mag == 0.0:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
            else:
                out_unit_x_f64[i] = dx / mag
                out_unit_y_f64[i] = dy / mag

    @njit(parallel=True, fastmath=True)
    def _alignment_from_neighbors_offsets(
        offsets_i64,
        neighbors_i32,
        headings_neighbors_f64,
        X_f64,
        Y_f64,
        x_vel_f64,
        y_vel_f64,
        sog_f64,
        consider_front_only_i8,
        out_unit_x_f64,
        out_unit_y_f64,
        out_mean_sog_f64,
        out_counts_i32,
    ):
        n = offsets_i64.shape[0] - 1
        for i in prange(n):
            start = offsets_i64[i]
            end = offsets_i64[i + 1]
            if start >= end:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
                out_mean_sog_f64[i] = np.nan
                out_counts_i32[i] = 0
                continue
            ax = X_f64[i]
            ay = Y_f64[i]
            vx_i = x_vel_f64[i]
            vy_i = y_vel_f64[i]
            sum_cos = 0.0
            sum_sin = 0.0
            sum_sog = 0.0
            cnt = 0
            for j in range(start, end):
                nb = neighbors_i32[j]
                if nb < 0:
                    continue
                if nb == i:
                    continue
                if consider_front_only_i8 != 0:
                    dx = X_f64[nb] - ax
                    dy = Y_f64[nb] - ay
                    if (dx * vx_i + dy * vy_i) <= 0.0:
                        continue
                h = headings_neighbors_f64[j]
                sum_cos += np.cos(h)
                sum_sin += np.sin(h)
                sum_sog += sog_f64[nb]
                cnt += 1
            out_counts_i32[i] = cnt
            if cnt <= 0:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
                out_mean_sog_f64[i] = np.nan
                continue
            mean_cos = sum_cos / cnt
            mean_sin = sum_sin / cnt
            mag = (mean_cos * mean_cos + mean_sin * mean_sin) ** 0.5
            if mag == 0.0:
                out_unit_x_f64[i] = 0.0
                out_unit_y_f64[i] = 0.0
            else:
                out_unit_x_f64[i] = mean_cos / mag
                out_unit_y_f64[i] = mean_sin / mag
            out_mean_sog_f64[i] = sum_sog / cnt

    @njit(parallel=True, fastmath=True)
    def _already_been_here_sparse_core_affine(
        agent_x_f64,
        agent_y_f64,
        hist_r_i16,
        hist_c_i16,
        hist_t_f32,
        hist_pos_i32,
        a_f64,
        e_f64,
        b_f64,
        c_f64,
        d_f64,
        f_f64,
        weight_f64,
        t_now_f64,
        horizon_f64,
        max_steps_i32,
        out_fx_f64,
        out_fy_f64,
    ):
        n = agent_x_f64.shape[0]
        K = hist_r_i16.shape[1]
        for i in prange(n):
            fx = 0.0
            fy = 0.0
            ax = agent_x_f64[i]
            ay = agent_y_f64[i]
            steps = K
            if max_steps_i32 > 0 and max_steps_i32 < K:
                steps = max_steps_i32

            # The sparse avoid history is a ring buffer; iterate from newest to
            # oldest so we can early-stop once entries fall outside the time
            # horizon or become uninitialized.
            pos = hist_pos_i32[i]
            if pos < 0:
                pos = 0
            pos = pos % K
            idx = pos - 1
            if idx < 0:
                idx += K

            for k in range(steps):
                j = idx - k
                if j < 0:
                    j += K

                rr = hist_r_i16[i, j]
                cc = hist_c_i16[i, j]
                tt = hist_t_f32[i, j]
                if rr < 0 or cc < 0 or (not np.isfinite(tt)):
                    # When the buffer is not yet full, older entries are
                    # uninitialized; since we're scanning oldestward, break.
                    break

                t_since = t_now_f64 - float(tt)
                # Scanning from newest->oldest means once we're outside the
                # horizon we can stop.
                if t_since >= horizon_f64:
                    break
                if t_since <= 10.0:
                    continue
                mult = 1.0 - (t_since - 5.0) / 7195.0
                wx = a_f64 * float(cc) + b_f64 * float(rr) + c_f64
                wy = d_f64 * float(cc) + e_f64 * float(rr) + f_f64
                dx = ax - wx
                dy = ay - wy
                dist2 = dx * dx + dy * dy
                if dist2 == 0.0:
                    dist2 = 1e-6
                fx += (weight_f64 * dx / dist2) * mult
                fy += (weight_f64 * dy / dist2) * mult
            out_fx_f64[i] = fx
            out_fy_f64[i] = fy
else:
    def _repulsive_core(agent_x, agent_y, world_x, world_y, multiplier, weight):
        # fallback Python implementation operating on flattened arrays with minimal temporaries
        wx = np.asarray(world_x).ravel()
        wy = np.asarray(world_y).ravel()
        mult = np.asarray(multiplier).ravel()
        dx = agent_x - wx
        dy = agent_y - wy
        mags = np.sqrt(dx * dx + dy * dy)
        mags[mags == 0] = 1e-6
        fx = ((weight * dx) / (mags * mags)) * mult
        fy = ((weight * dy) / (mags * mags)) * mult
        return float(np.nansum(fx)), float(np.nansum(fy))

    def _repulsive_batched_python(agent_xs, agent_ys, mmap_flat, mmap_offsets, rows_min, cols_min, nr_list, nc_list, affine, weight, t):
        # Python fallback: compute multiplier and world coords on the fly
        a = affine[0]
        b = affine[1]
        c = affine[2]
        d = affine[3]
        e = affine[4]
        f = affine[5]
        n_agents = int(agent_xs.shape[0])
        out_x = np.empty(n_agents, dtype=np.float64)
        out_y = np.empty(n_agents, dtype=np.float64)
        for ai in range(n_agents):
            ax = float(agent_xs[ai])
            ay = float(agent_ys[ai])
            start = int(mmap_offsets[ai])
            nr = int(nr_list[ai])
            nc = int(nc_list[ai])
            if nr <= 0 or nc <= 0:
                out_x[ai] = 0.0
                out_y[ai] = 0.0
                continue
            tx = 0.0
            ty = 0.0
            for ri in range(nr):
                for ci in range(nc):
                    idx = start + ri * nc + ci
                    val = mmap_flat[idx]
                    t_since = t - val
                    m = 0.0
                    if t_since > 10.0 and t_since < 7200.0:
                        m = 1.0 - (t_since - 5.0) / 7195.0
                    row = rows_min[ai] + ri
                    col = cols_min[ai] + ci
                    wx = a * col + b * row + c
                    wy = d * col + e * row + f
                    dx = ax - wx
                    dy = ay - wy
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag == 0.0:
                        mag = 1e-6
                    ux = dx / mag
                    uy = dy / mag
                    fx = ((weight * ux) / mag) * m
                    fy = ((weight * uy) / mag) * m
                    tx += fx
                    ty += fy
            out_x[ai] = tx
            out_y[ai] = ty
        return out_x, out_y


class behavior():
    def __init__(self, dt, simulation_object):
        self.dt = dt
        self.simulation = simulation_object
        # Async diagnostics queue and thread
        self._diag_queue = None
        self._diag_thread = None
        self._diag_thread_running = False
        # Preallocated buffers for batched repulsive computation
        self._buf_world_x = None
        self._buf_world_y = None
        self._buf_mult = None
        self._buf_offsets = None
        self._buf_counts = None
        self._buf_rows_min = None
        self._buf_cols_min = None
        self._buf_nr = None
        self._buf_nc = None
        # reusable output buffer and scratch arrays to avoid repeated allocations
        self._repulsive_out = None
        self._scratch_counts = None
        self._scratch_offsets = None
        # per-batch logging entries: list of dicts with keys 'start','end','time','rss_before','rss_after','batch_total'
        self._batch_log = []
        # threshold in bytes to consider reducing batch size (default 200MB)
        self._rss_threshold_bytes = int(getattr(self.simulation, 'behavior_memory_threshold_bytes', 200 * 1024 * 1024))
        # small cache to reuse world grids for repeated windows
        self._world_grid_cache = getattr(self, '_world_grid_cache', {})
        # index / meshgrid cache keyed by (nr,nc)
        self._meshgrid_cache = {}
        # cached psutil.Process instance (initialized lazily)
        self._psutil_proc = None
        # cached output buffers for batched kernel to avoid per-call allocations
        self._out_x = None
        self._out_y = None
        # Optional pre-sizing knobs: allow environments to set conservative maxima to avoid resizing
        try:
            import os as _os
            max_agents = int(getattr(self.simulation, 'num_agents', int(_os.environ.get('SIM_MAX_AGENTS', 20000))))
            max_window = int(getattr(self.simulation, 'behavior_max_window', int(_os.environ.get('BEHAVIOR_MAX_WINDOW', 441))))
            # preallocate common scratch arrays to avoid repeated allocations at runtime
            try:
                self._scratch_offsets = np.empty(max_agents, dtype=np.int64)
                self._scratch_counts = np.empty(max_agents, dtype=np.int64)
                self._repulsive_out = np.empty((max_agents, 2), dtype=np.float64)
                # preallocate a per-batch timestamp buffer sized by the maximum window
                self._buf_mult = np.empty(max_window, dtype=np.float64)
                # preallocate per-agent per-batch descriptor buffers for a conservative batch size
                conservative_batch = min(1024, max_agents)
                self._buf_offsets = np.empty(conservative_batch, dtype=np.int64)
                self._buf_counts = np.empty(conservative_batch, dtype=np.int64)
                self._buf_rows_min = np.empty(conservative_batch, dtype=np.int64)
                self._buf_cols_min = np.empty(conservative_batch, dtype=np.int64)
                self._buf_nr = np.empty(conservative_batch, dtype=np.int64)
                self._buf_nc = np.empty(conservative_batch, dtype=np.int64)
                self._out_x = np.empty(conservative_batch, dtype=np.float64)
                self._out_y = np.empty(conservative_batch, dtype=np.float64)
            except Exception:
                # fall back to lazy allocation if memory allocation fails
                pass
        except Exception:
            pass
        # set numba threads to CPU count if available and not already set via env
        try:
            if _NUMBA_AVAILABLE:
                import os
                # Initialize Numba configuration and warm JIT kernels once per process.
                try:
                    if not getattr(self.__class__, '_numba_initialized', False):
                        if 'NUMBA_NUM_THREADS' not in os.environ:
                            try:
                                from multiprocessing import cpu_count
                                from numba import set_num_threads
                                set_num_threads(cpu_count())
                            except Exception:
                                pass
                        try:
                            self._warmup_numba_kernels()
                        except Exception:
                            pass
                        try:
                            setattr(self.__class__, '_numba_initialized', True)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _safe_npz_dump(self, outdir, fname_prefix, payload):
        """Write a compressed NPZ of `payload` to `outdir` with `fname_prefix`.
        Best-effort: failures are swallowed and None returned on error.
        """
        try:
            # If async diag thread is running, enqueue the payload and return immediately
            if self._diag_queue is not None and self._diag_thread_running:
                try:
                    self._enqueue_diag(outdir, fname_prefix, payload)
                    return None
                except Exception:
                    pass
            import numpy as _np
            import os, time
            os.makedirs(outdir, exist_ok=True)
            ts = int(time.time())
            fname = os.path.join(outdir, f"{fname_prefix}_{ts}.npz")
            ser = {k: _np.asarray(v).astype(float) for k, v in payload.items()}
            _np.savez_compressed(fname, **ser)
            return fname
        except Exception:
            return None

    def _diag_worker(self):
        # Simple robust diagnostics writer: pop queue items and write JSON or NPZ.
        import time, os, json
        import numpy as _np
        logger = logging.getLogger(__name__)
        while self._diag_thread_running:
            try:
                item = self._diag_queue.get(timeout=0.5)
            except Exception:
                continue
            try:
                outdir, fname_prefix, payload = item
                try:
                    os.makedirs(outdir, exist_ok=True)
                except Exception:
                    logger.debug('diag: failed to make outdir %s', outdir, exc_info=True)
                ts = int(time.time())
                # JSON payload requested
                if isinstance(payload, dict) and payload.get('_fmt') == 'json':
                    fname = os.path.join(outdir, f"{fname_prefix}_{ts}.json")
                    try:
                        with open(fname, 'w', encoding='utf-8') as fh:
                            json.dump(payload.get('obj', {}), fh, indent=2)
                    except Exception:
                        logger.debug('diag: failed to write json %s', fname, exc_info=True)
                else:
                    # default: compressed NPZ of numeric-ish content
                    fname = os.path.join(outdir, f"{fname_prefix}_{ts}.npz")
                    try:
                        ser = {}
                        if isinstance(payload, dict):
                            for k, v in payload.items():
                                try:
                                    ser[k] = _np.asarray(v).astype(float)
                                except Exception:
                                    # try scalar conversion
                                    try:
                                        ser[k] = float(v)
                                    except Exception:
                                        pass
                        # write at least something
                        if ser:
                            _np.savez_compressed(fname, **ser)
                        else:
                            # fallback: write a tiny marker file so callers see an artifact
                            with open(os.path.join(outdir, f"{fname_prefix}_{ts}.marker"), 'w', encoding='utf-8') as fh:
                                fh.write('empty')
                    except Exception:
                        logger.debug('diag: failed to write npz %s', fname, exc_info=True)
            finally:
                try:
                    self._diag_queue.task_done()
                except Exception:
                    logger.debug('diag: task_done failed', exc_info=True)

    def _start_diag_thread(self):
        if self._diag_thread is not None and self._diag_thread_running:
            return
        self._diag_queue = queue.Queue()
        self._diag_thread_running = True
        self._diag_thread = threading.Thread(target=self._diag_worker, daemon=True)
        self._diag_thread.start()

    def _stop_diag_thread(self):
        if self._diag_thread is None:
            return
        self._diag_thread_running = False
        try:
            self._diag_thread.join(timeout=1.0)
        except Exception:
            pass

    def _enqueue_diag(self, outdir, fname_prefix, payload):
        if self._diag_queue is None:
            self._start_diag_thread()
        self._diag_queue.put((outdir, fname_prefix, payload))

    def _enqueue_diag_json(self, outdir, fname_prefix, obj):
        if self._diag_queue is None:
            self._start_diag_thread()
        payload = {'_fmt': 'json', 'obj': obj}
        self._diag_queue.put((outdir, fname_prefix, payload))

    def _safe_write_diagnostics(self, step_i, payload, outdir=None):
        """Attempt to write diagnostics via diagnostics_writer, falling back to NPZ.
        Returns True if HDF5 writer succeeded, False otherwise.
        """
        dw = getattr(self.simulation, 'diagnostics_writer', None)
        # prefer the HDF5 diagnostics writer
        if dw is not None:
            try:
                dw.write_step(step_i, payload)
                return True
            except Exception:
                # try NPZ fallback when explicitly requested
                pass

        if outdir is None:
            outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
        if getattr(self.simulation, 'force_npz_fallback', False):
            try:
                self._safe_npz_dump(outdir, f'behavior_debug_guarded_step_{step_i}', payload)
            except Exception:
                pass
        return False

    def _safe_set_sim_attr(self, name, value):
        """Set attribute `name` on simulation in a best-effort way.
        Converts numpy arrays to native types where possible. Swallows exceptions.
        """
        try:
            setattr(self.simulation, name, value)
        except Exception:
            try:
                self.simulation.__dict__[name] = value
            except Exception:
                pass

    def _safe_asarray(self, v, dtype=float, default=None):
        """Return np.asarray(v, dtype) or `default` on failure."""
        try:
            return np.asarray(v, dtype=dtype)
        except Exception:
            return default

    def _get_env(self, key: str, default=None):
        """Get an environment dataset, using simulation caching when available."""
        sim = self.simulation
        try:
            key = str(key)
        except Exception:
            return default
        try:
            getter = getattr(sim, 'get_cached_dataset', None)
            if callable(getter) and key.startswith('environment/'):
                return getter(key, default=default)
        except Exception:
            pass
        h5 = hdf5_io.get_hdf5_obj(sim)
        return hdf5_io.read_dataset(h5, key, default=default)

    def _window_rc(self, rows: np.ndarray, cols: np.ndarray, buff: int, shape: tuple[int, int]):
        """Build per-agent window indices (rows, cols) and validity mask.

        Returns `(rr_clip, cc_clip, valid)` each shaped (n_agents, 2*buff+1, 2*buff+1).
        """
        rows = np.asarray(rows, dtype=np.int32).reshape((-1,))
        cols = np.asarray(cols, dtype=np.int32).reshape((-1,))
        H, W = int(shape[0]), int(shape[1])
        buff = int(buff)
        if buff < 0:
            buff = 0

        if not hasattr(self, "_offset_cache"):
            self._offset_cache = {}
        offs = self._offset_cache.get(buff)
        if offs is None:
            offs = np.arange(-buff, buff + 1, dtype=np.int32)
            self._offset_cache[buff] = offs

        rr = rows[:, np.newaxis, np.newaxis] + offs[np.newaxis, :, np.newaxis]
        cc = cols[:, np.newaxis, np.newaxis] + offs[np.newaxis, np.newaxis, :]
        valid = (rr >= 0) & (cc >= 0) & (rr < H) & (cc < W)
        rr_clip = np.clip(rr, 0, max(0, H - 1)).astype(np.int32, copy=False)
        cc_clip = np.clip(cc, 0, max(0, W - 1)).astype(np.int32, copy=False)
        return rr_clip, cc_clip, valid

    def _window_dxdy(self, rr: np.ndarray, cc: np.ndarray, transform, agent_x: np.ndarray, agent_y: np.ndarray):
        """Compute dx,dy from agent positions to window cell centers."""
        a, b, c, d, e, f = _unpack_affine(transform)
        x_cell = float(a) * cc + float(b) * rr + float(c)
        y_cell = float(d) * cc + float(e) * rr + float(f)
        dx = x_cell - agent_x[:, np.newaxis, np.newaxis]
        dy = y_cell - agent_y[:, np.newaxis, np.newaxis]
        return dx, dy

    def _coerce_agent_vec(self, vec) -> np.ndarray:
        """Coerce cue output to an array of shape (num_agents, 2)."""
        arr = np.asarray(vec)
        n = int(getattr(self.simulation, 'num_agents', 0) or 0)

        if n > 0:
            try:
                if arr.shape == (n, 2):
                    return arr
            except Exception:
                pass

            if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == n:
                return arr.T
            if arr.ndim == 1 and arr.size == 2:
                return np.tile(arr, (n, 1))
            if arr.ndim == 1 and arr.size == n:
                return np.column_stack((arr, np.zeros(n)))
            if arr.ndim >= 2:
                axes = [i for i, s in enumerate(arr.shape) if s == n]
                if axes:
                    axis = axes[0]
                    moved = np.moveaxis(arr, axis, 0)
                    collapsed = np.nan_to_num(moved).reshape(n, -1).sum(axis=1)
                    return np.column_stack((collapsed, np.zeros(n)))

        return np.zeros((n, 2), dtype=float) if n > 0 else np.zeros((0, 2), dtype=float)

    def _record_cue_shapes(self, cue_dict: dict, t) -> None:
        shapes = {}
        for k, v in (cue_dict or {}).items():
            try:
                arr = np.asarray(v)
                shapes[str(k)] = {'ndim': int(arr.ndim), 'shape': tuple(arr.shape)}
            except Exception:
                shapes[str(k)] = {'error': 'cannot convert to array'}

        self._safe_set_sim_attr('cue_shapes', shapes)

        if getattr(self.simulation, 'debug_behavior', False):
            outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
            step_i = int(getattr(self.simulation, 'current_step', t))
            try:
                self._enqueue_diag_json(outdir, f'cue_shapes_{step_i}', shapes)
            except Exception:
                pass

    def _rawvecs_payload(self, raw_vecs: dict, cue_magnitudes: dict) -> dict:
        payload = {}
        for k, v in (raw_vecs or {}).items():
            try:
                payload[f'{k}_vec'] = np.asarray(v).astype(float)
            except Exception:
                pass
        for k, v in (cue_magnitudes or {}).items():
            try:
                payload[f'{k}_mag'] = np.asarray(v).astype(float)
            except Exception:
                pass

        try:
            # Prefer CSR neighbors when available (avoids list-of-arrays construction).
            offs = getattr(self.simulation, 'neighbors_offsets', None)
            inds = getattr(self.simulation, 'neighbors_indices', None)
            if offs is not None and inds is not None:
                offs = np.asarray(offs, dtype=np.int64).reshape((-1,))
                if offs.size >= 2:
                    counts = (offs[1:] - offs[:-1]).astype(np.int32, copy=False)
                    payload['neighbor_counts'] = counts
                    if int(np.sum(counts)) > 0:
                        payload['neighbors_concat'] = np.asarray(inds, dtype=np.int32).reshape((-1,))
            elif hasattr(self.simulation, 'agents_within_buffers'):
                neighbor_counts = np.array([len(x) for x in self.simulation.agents_within_buffers], dtype=np.int32)
                payload['neighbor_counts'] = neighbor_counts
                if neighbor_counts.sum() > 0:
                    try:
                        payload['neighbors_concat'] = np.concatenate(self.simulation.agents_within_buffers).astype(np.int32)
                    except Exception:
                        payload['neighbors_concat'] = np.array([], dtype=np.int32)
        except Exception:
            pass

        if not payload:
            payload['marker'] = np.array([1], dtype=np.int8)
        return payload

    def _maybe_dump_rawvecs(self, t, raw_vecs: dict, cue_magnitudes: dict) -> None:
        outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
        step_i = int(getattr(self.simulation, 'current_step', t))

        if getattr(self.simulation, 'debug_behavior', False):
            payload = self._rawvecs_payload(raw_vecs, cue_magnitudes)
            try:
                self._safe_npz_dump(outdir, f'behavior_debug_rawvecs_step_{step_i}', payload)
            except Exception:
                pass

        if os.environ.get('FORCE_RAWVECS', '').lower() == 'true':
            payload = self._rawvecs_payload(raw_vecs, cue_magnitudes)
            try:
                self._safe_npz_dump(outdir, f'behavior_debug_rawvecs_FORCE_step_{step_i}', payload)
            except Exception as e:
                try:
                    logging.getLogger(__name__).debug('FORCE_RAWVECS failed to write NPZ: %s', e)
                except Exception:
                    pass

    def _warmup_numba_kernels(self):
        # Call numba kernels with tiny dummy data to force compilation ahead of timed runs
        if not _NUMBA_AVAILABLE:
            return
        try:
            import numpy as _np
            # tiny arrays
            axs = _np.array([0.0, 1.0], dtype=_np.float64)
            ays = _np.array([0.0, 1.0], dtype=_np.float64)
            mmap_flat = _np.array([0.0, 0.0], dtype=_np.float64)
            mmap_offsets = _np.array([0, 1], dtype=_np.int64)
            rows_min = _np.array([0, 0], dtype=_np.int64)
            cols_min = _np.array([0, 0], dtype=_np.int64)
            nr = _np.array([1, 1], dtype=_np.int64)
            nc = _np.array([1, 1], dtype=_np.int64)
            affine = _np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=_np.float64)
            try:
                _repulsive_batched_core_safe(axs, ays, mmap_flat, mmap_offsets, rows_min, cols_min, nr, nc, affine, 1.0, 0.0)
            except Exception:
                pass
            try:
                _repulsive_core_safe(0.0, 0.0, _np.array([0.0]), _np.array([0.0]), _np.array([1.0]), 1.0)
            except Exception:
                pass
            try:
                ar = _np.array([0, 1], dtype=_np.int32)
                ac = _np.array([0, 1], dtype=_np.int32)
                hr = _np.array([[0, -1], [1, -1]], dtype=_np.int16)
                hc = _np.array([[1, -1], [0, -1]], dtype=_np.int16)
                ht = _np.array([[0.0, _np.nan], [0.0, _np.nan]], dtype=_np.float32)
                hp = _np.array([0, 0], dtype=_np.int32)
                ax = _np.array([0.0, 1.0], dtype=_np.float64)
                ay = _np.array([0.0, 1.0], dtype=_np.float64)
                out_fx = _np.zeros(2, dtype=_np.float64)
                out_fy = _np.zeros(2, dtype=_np.float64)
                _already_been_here_sparse_core_affine(ax, ay, hr, hc, ht, hp, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 7200.0, 1, out_fx, out_fy)
            except Exception:
                pass
            try:
                # Warm up schooling kernels (cohesion/alignment) with a tiny graph.
                offsets = _np.array([0, 1, 2], dtype=_np.int64)  # 2 agents, 1 neighbor each
                neighbors = _np.array([1, 0], dtype=_np.int32)
                X = _np.array([0.0, 1.0], dtype=_np.float64)
                Y = _np.array([0.0, 0.0], dtype=_np.float64)
                x_vel = _np.array([1.0, 1.0], dtype=_np.float64)
                y_vel = _np.array([0.0, 0.0], dtype=_np.float64)
                headings = _np.array([0.0, 0.0], dtype=_np.float64)
                headings_neighbors = headings[neighbors].astype(_np.float64)
                sog = _np.array([0.2, 0.2], dtype=_np.float64)
                outx = _np.zeros(2, dtype=_np.float64)
                outy = _np.zeros(2, dtype=_np.float64)
                _cohesion_from_neighbors_offsets(offsets, neighbors, X, Y, x_vel, y_vel, _np.int8(0), outx, outy)
                mean_sog = _np.full(2, _np.nan, dtype=_np.float64)
                counts = _np.zeros(2, dtype=_np.int32)
                _alignment_from_neighbors_offsets(offsets, neighbors, headings_neighbors, X, Y, x_vel, y_vel, sog, _np.int8(0), outx, outy, mean_sog, counts)
            except Exception:
                pass
        except Exception:
            pass

    def _get_psutil_proc(self):
        """Lazily create and cache a psutil.Process() object for repeated sampling."""
        if not _PSUTIL_AVAILABLE:
            return None
        if getattr(self, '_psutil_proc', None) is None:
            try:
                self._psutil_proc = psutil.Process()
            except Exception:
                self._psutil_proc = None
        return getattr(self, '_psutil_proc', None)

    def _already_been_here_sparse(self, weight: float, t: float):
        """Sparse avoid-memory variant of `already_been_here`.

        Uses per-agent ring-buffer history (`simulation.avoid_hist_*`) instead of
        dense per-agent HDF5 rasters, avoiding per-agent dataset reads and writes.
        Returns None when sparse history is unavailable.
        """
        sim = self.simulation
        rows_hist = getattr(sim, 'avoid_hist_rows', None)
        cols_hist = getattr(sim, 'avoid_hist_cols', None)
        t_hist = getattr(sim, 'avoid_hist_t', None)
        if rows_hist is None or cols_hist is None or t_hist is None:
            return None

        # Avoid needless conversions on the hot path: these are typically already
        # numpy arrays allocated by `simulation.ensure_avoid_history()`.
        if not isinstance(rows_hist, np.ndarray):
            try:
                rows_hist = np.asarray(rows_hist)
            except Exception:
                return None
        if not isinstance(cols_hist, np.ndarray):
            try:
                cols_hist = np.asarray(cols_hist)
            except Exception:
                return None
        if not isinstance(t_hist, np.ndarray):
            try:
                t_hist = np.asarray(t_hist)
            except Exception:
                return None

        n = int(getattr(sim, 'num_agents', 0) or 0)
        if n <= 0:
            return None
        if rows_hist.ndim != 2 or cols_hist.ndim != 2 or t_hist.ndim != 2:
            return None
        if rows_hist.shape[0] != n or cols_hist.shape[0] != n or t_hist.shape[0] != n:
            return None
        if rows_hist.shape != cols_hist.shape or rows_hist.shape != t_hist.shape:
            return None

        try:
            affine = _unpack_affine(getattr(sim, 'mental_map_transform', getattr(sim, 'depth_rast_transform', None)))
        except Exception:
            return None
        a, b, c, d, e, f = [float(x) for x in affine]

        agent_x = np.nan_to_num(np.asarray(sim.X, dtype=float)).reshape((-1,))
        agent_y = np.nan_to_num(np.asarray(sim.Y, dtype=float)).reshape((-1,))
        if agent_x.size != n or agent_y.size != n:
            return None

        horizon = float(getattr(sim, 'avoid_memory_horizon_s', 7200.0))
        # Cap how many most-recent entries we examine per agent when computing
        # avoid force. This keeps the per-step cost bounded even when the ring
        # buffer is large.
        max_steps = int(getattr(sim, 'avoid_force_history_len', 0) or 0)

        t_now = float(t)
        fx = np.zeros(n, dtype=np.float64)
        fy = np.zeros(n, dtype=np.float64)

        # Prefer the Numba kernel when available: avoids allocating (n_agents x chunk)
        # temporaries and keeps results aligned with world coordinates.
        if _NUMBA_AVAILABLE:
            try:
                ax = np.ascontiguousarray(agent_x, dtype=np.float64)
                ay = np.ascontiguousarray(agent_y, dtype=np.float64)
                hr = np.ascontiguousarray(rows_hist, dtype=np.int16)
                hc = np.ascontiguousarray(cols_hist, dtype=np.int16)
                ht = np.ascontiguousarray(t_hist, dtype=np.float32)
                hp = np.ascontiguousarray(getattr(sim, 'avoid_hist_pos', np.zeros(n, dtype=np.int32)), dtype=np.int32).reshape((-1,))
                if hp.size != n:
                    hp = np.zeros(n, dtype=np.int32)
                _already_been_here_sparse_core_affine(
                    ax,
                    ay,
                    hr,
                    hc,
                    ht,
                    hp,
                    float(a),
                    float(e),
                    float(b),
                    float(c),
                    float(d),
                    float(f),
                    float(weight),
                    float(t_now),
                    float(horizon),
                    int(max_steps),
                    fx,
                    fy,
                )
                return np.column_stack((fx, fy))
            except Exception:
                # Fall back to the vectorized implementation below.
                pass

        # Fallback when Numba isn't available: iterate the ring buffer from
        # newest to oldest, with the same early-stop semantics as the JIT kernel.
        try:
            pos = np.asarray(getattr(sim, 'avoid_hist_pos', np.zeros(n, dtype=int)), dtype=int).reshape((-1,))
        except Exception:
            pos = np.zeros(n, dtype=int)
        if pos.size != n:
            pos = np.zeros(n, dtype=int)

        K = int(rows_hist.shape[1])
        steps = K
        if max_steps > 0 and max_steps < K:
            steps = int(max_steps)

        for i in range(n):
            p = int(pos[i])
            if p < 0:
                p = 0
            p = p % K
            idx = p - 1
            if idx < 0:
                idx += K
            ax = float(agent_x[i])
            ay = float(agent_y[i])
            for k in range(steps):
                j = idx - k
                if j < 0:
                    j += K
                rr = int(rows_hist[i, j])
                cc = int(cols_hist[i, j])
                tt = float(t_hist[i, j])
                if rr < 0 or cc < 0 or (not np.isfinite(tt)):
                    break
                t_since = float(t_now) - tt
                if t_since >= float(horizon):
                    break
                if t_since <= 10.0:
                    continue
                mult = 1.0 - (t_since - 5.0) / 7195.0
                wx = float(a) * float(cc) + float(b) * float(rr) + float(c)
                wy = float(d) * float(cc) + float(e) * float(rr) + float(f)
                dx = ax - wx
                dy = ay - wy
                dist2 = dx * dx + dy * dy
                if dist2 == 0.0:
                    dist2 = 1e-6
                fx[i] += (float(weight) * dx / dist2) * mult
                fy[i] += (float(weight) * dy / dist2) * mult

        return np.column_stack((fx, fy))

    def already_been_here(self, weight, t):
        try:
            if float(weight) == 0.0:
                return np.zeros((self.simulation.num_agents, 2), dtype=float)
        except Exception:
            pass

        # Fast path: sparse avoid history (no per-agent HDF5 memory rasters).
        try:
            if bool(getattr(self.simulation, 'use_sparse_avoid_memory', False)):
                out = self._already_been_here_sparse(weight=float(weight), t=float(t))
                if out is not None:
                    return out
        except Exception:
            pass

        x, y = np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y)

        # use the mental map transform (coarser avoid-cell grid) when converting
        # geographic positions to memory pixel indices. Previously the depth
        # raster transform was used which produced indices on a different grid
        # and resulted in out-of-bounds / empty slices causing zero forces.
        mental_map_rows, mental_map_cols = geo_to_pixel(x, y, getattr(self.simulation, 'mental_map_transform', getattr(self.simulation, 'depth_rast_transform', None)))
        # Ensure indices are 1-D integer arrays even for single-agent scalar inputs
        try:
            mental_map_rows = np.atleast_1d(mental_map_rows).astype(int)
            mental_map_cols = np.atleast_1d(mental_map_cols).astype(int)
        except Exception:
            mental_map_rows = np.array([int(mental_map_rows)])
            mental_map_cols = np.array([int(mental_map_cols)])

        buff = 10
        row_min = np.clip(mental_map_rows - buff, 0, None)
        # use hdf5_io to support both h5py.File and dict-like mocks
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        memory0 = hdf5_io.read_dataset(h5, 'memory/0', default=np.zeros((1, 1)))
        row_max = np.clip(mental_map_rows + buff + 1, None, memory0.shape[0])
        col_min = np.clip(mental_map_cols - buff, 0, None)
        col_max = np.clip(mental_map_cols + buff + 1, None, memory0.shape[1])

        # cache HDF5-like object to avoid repeated opens and batch-read memory per-agent
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        # Build concatenated buffers for all agents' windows to feed batched kernel
        agent_count = int(self.simulation.num_agents)
        agent_xs = np.nan_to_num(self.simulation.X).astype(np.float64)
        agent_ys = np.nan_to_num(self.simulation.Y).astype(np.float64)

        # First pass: determine total size to preallocate concatenated buffers
        if self._scratch_offsets is None or self._scratch_offsets.shape[0] < agent_count:
            self._scratch_offsets = np.empty(agent_count, dtype=np.int64)
        if self._scratch_counts is None or self._scratch_counts.shape[0] < agent_count:
            self._scratch_counts = np.empty(agent_count, dtype=np.int64)
        offsets = self._scratch_offsets[:agent_count]
        counts = self._scratch_counts[:agent_count]
        total_elems = 0
        mem_sections = [None] * agent_count
        rows_list = [None] * agent_count
        cols_list = [None] * agent_count
        # cached empty window to avoid repeated allocation of np.zeros((1,1)) when datasets are missing
        if not hasattr(self, '_empty_window'):
            self._empty_window = np.empty((1, 1), dtype=np.float64)

        for a in range(agent_count):
            rmin = int(row_min[a])
            rmax = int(row_max[a])
            cmin = int(col_min[a])
            cmax = int(col_max[a])
            try:
                mmap = hdf5_io.read_dataset(h5, f'memory/{a}', default=None)
                if mmap is None:
                    mmap = self._empty_window
            except Exception:
                mmap = self._empty_window
            section = mmap[rmin:rmax, cmin:cmax]
            # flatten and ensure float64 contiguous to avoid repeated ravel/astype later
            sec_flat = section.ravel()
            if sec_flat.dtype == np.float64 and sec_flat.flags['C_CONTIGUOUS']:
                mem_sections[a] = sec_flat
            else:
                mem_sections[a] = np.ascontiguousarray(sec_flat, dtype=np.float64)
            nr = rmax - rmin
            nc = cmax - cmin
            cnt = int(mem_sections[a].size)
            counts[a] = cnt
            offsets[a] = total_elems
            total_elems += cnt
            rows_list[a] = (rmin, rmax)
            cols_list[a] = (cmin, cmax)

        if total_elems == 0:
            # nothing to compute
            if self._repulsive_out is None or self._repulsive_out.shape[0] < agent_count:
                self._repulsive_out = np.empty((agent_count, 2), dtype=np.float64)
            repulsive_forces_per_agent = self._repulsive_out[:agent_count]
        else:
            # Process agents in manageable batches to avoid a single huge allocation
            batch_size = int(getattr(self.simulation, 'behavior_batch_size', 256))
            if self._repulsive_out is None or self._repulsive_out.shape[0] < agent_count:
                self._repulsive_out = np.zeros((agent_count, 2), dtype=np.float64)
            repulsive_out = self._repulsive_out[:agent_count]
            try:
                # Cache psutil.Process() and sample RSS once per-method to reduce sampling overhead
                ps_proc = self._get_psutil_proc()
                rss_before_method = None
                rss_after_method = None
                try:
                    if ps_proc is not None:
                        rss_before_method = ps_proc.memory_info().rss
                except Exception:
                    rss_before_method = None

                # precompute affine once to avoid repeated unpacking and conversions
                try:
                    affine = _unpack_affine(getattr(self.simulation, 'mental_map_transform', None))
                except Exception:
                    affine = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

                for bstart in range(0, agent_count, batch_size):
                    bend = min(agent_count, bstart + batch_size)
                    batch_counts = counts[bstart:bend]
                    # compute elements needed for this batch
                    batch_total = int(batch_counts.sum())
                    if batch_total == 0:
                        continue
                    # per-batch sampling disabled — use per-method rss_before_method (stored in batch log below)
                    rss_before = rss_before_method
                    # ensure buffers large enough for batch: mmap_flat will hold timestamps
                    if self._buf_mult is None or self._buf_mult.shape[0] < batch_total:
                        self._buf_mult = np.empty(batch_total, dtype=np.float64)
                    nb = bend - bstart
                    # per-agent descriptor arrays (offsets/counts) must be sized by number of agents in batch
                    if self._buf_offsets is None or self._buf_offsets.shape[0] < nb:
                        self._buf_offsets = np.empty(nb, dtype=np.int64)
                    if self._buf_counts is None or self._buf_counts.shape[0] < nb:
                        self._buf_counts = np.empty(nb, dtype=np.int64)

                    write_ptr = 0
                    local_offsets = self._buf_offsets[:nb]
                    local_counts = self._buf_counts[:nb]
                    # per-agent row/col/nr/nc buffers
                    if self._buf_rows_min is None or self._buf_rows_min.shape[0] < nb:
                        self._buf_rows_min = np.empty(nb, dtype=np.int64)
                        self._buf_cols_min = np.empty(nb, dtype=np.int64)
                        self._buf_nr = np.empty(nb, dtype=np.int64)
                        self._buf_nc = np.empty(nb, dtype=np.int64)
                    local_rows_min = self._buf_rows_min[:nb]
                    local_cols_min = self._buf_cols_min[:nb]
                    local_nr = self._buf_nr[:nb]
                    local_nc = self._buf_nc[:nb]
                    for i, a in enumerate(range(bstart, bend)):
                        cnt = int(counts[a])
                        local_counts[i] = cnt
                        if cnt == 0:
                            local_offsets[i] = 0
                            # Ensure all per-agent descriptors are initialized so the
                            # batched kernel never reads uninitialized values.
                            local_rows_min[i] = 0
                            local_cols_min[i] = 0
                            local_nr[i] = 0
                            local_nc[i] = 0
                            continue
                        rmin, rmax = rows_list[a]
                        cmin, cmax = cols_list[a]
                        section = mem_sections[a]
                        # store raw timestamp values into mmap_flat; prefer a contiguous float64 view when possible
                        sec_ravel = section.ravel()
                        if sec_ravel.dtype == np.float64 and sec_ravel.flags['C_CONTIGUOUS']:
                            tvals = sec_ravel
                        else:
                            tvals = np.ascontiguousarray(sec_ravel, dtype=np.float64)
                        nr = rmax - rmin
                        nc = cmax - cmin
                        # reuse arange and meshgrid buffers when possible
                        key = (nr, nc)
                        try:
                            if key in self._meshgrid_cache:
                                col_grid, row_grid = self._meshgrid_cache[key]
                            else:
                                rows_idx = np.arange(rmin, rmax)
                                cols_idx = np.arange(cmin, cmax)
                                col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)
                                # store contiguous copies
                                self._meshgrid_cache[key] = (np.ascontiguousarray(col_grid), np.ascontiguousarray(row_grid))
                        except Exception:
                            rows_idx = np.arange(rmin, rmax)
                            cols_idx = np.arange(cmin, cmax)
                            col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)
                        # copy timestamps to mmap_flat buffer
                        actual_cnt = int(tvals.size)
                        if actual_cnt != cnt:
                            # defensive: if the underlying section size differs from the
                            # precomputed count, use the actual size and update counters
                            cnt = actual_cnt
                        if cnt > 0:
                            self._buf_mult[write_ptr:write_ptr+cnt] = tvals[:cnt]
                        local_offsets[i] = write_ptr
                        local_counts[i] = cnt
                        local_rows_min[i] = rmin
                        local_cols_min[i] = cmin
                        local_nr[i] = nr
                        local_nc[i] = nc
                        write_ptr += cnt

                    # per-batch sampling disabled — use per-method rss_after_method (computed after loop)
                    rss_after = None

                    # log batch
                    try:
                        run_tag = getattr(self.simulation, 'run_tag', None) or os.environ.get('RUN_TAG')
                        self._batch_log.append({'start': bstart, 'end': bend, 'time': None, 'rss_before': rss_before, 'rss_after': rss_after, 'batch_total': batch_total, 'n_agents': int(agent_count), 'batch_size': int(batch_size), 'run_tag': run_tag})
                    except Exception:
                        pass

                    # Call batched kernel for this batch
                    axs = agent_xs[bstart:bend]
                    ays = agent_ys[bstart:bend]
                    try:
                        # time the kernel call
                        t0 = time.time()
                        if _NUMBA_AVAILABLE:
                            # ensure out buffers are large enough
                            if self._out_x is None or self._out_x.shape[0] < nb:
                                self._out_x = np.empty(nb, dtype=np.float64)
                                self._out_y = np.empty(nb, dtype=np.float64)
                            # If the low-level compiled function is available, call it with preallocated outputs
                            if '_repulsive_batched_core' in globals():
                                mf = np.ascontiguousarray(self._buf_mult[:write_ptr], dtype=np.float64)
                                mo = np.ascontiguousarray(local_offsets, dtype=np.int64)
                                rm = np.ascontiguousarray(local_rows_min, dtype=np.int64)
                                cm = np.ascontiguousarray(local_cols_min, dtype=np.int64)
                                nr_arr = np.ascontiguousarray(local_nr, dtype=np.int64)
                                nc_arr = np.ascontiguousarray(local_nc, dtype=np.int64)
                                axs_c = np.ascontiguousarray(axs, dtype=np.float64)
                                ays_c = np.ascontiguousarray(ays, dtype=np.float64)
                                aff_c = np.ascontiguousarray(affine, dtype=np.float64)
                                _repulsive_batched_core(axs_c, ays_c, mf, mo, rm, cm, nr_arr, nc_arr, aff_c, float(weight), float(t), self._out_x[:nb], self._out_y[:nb])
                                out_x = self._out_x[:nb]
                                out_y = self._out_y[:nb]
                            else:
                                out_x, out_y = _repulsive_batched_core_safe(axs, ays, self._buf_mult[:write_ptr], local_offsets, local_rows_min, local_cols_min, local_nr, local_nc, affine, float(weight), float(t))
                        else:
                            # Python fallback expects world coords; reuse output buffers to avoid repeated allocations
                            out_x = np.empty(nb, dtype=np.float64)
                            out_y = np.empty(nb, dtype=np.float64)
                            for idx, a in enumerate(range(bstart, bend)):
                                try:
                                    out_x[idx], out_y[idx] = self._calculate_repulsive_force(mem_sections[a], a, int(row_min[a]), int(row_max[a]), int(col_min[a]), int(col_max[a]), weight, t)
                                except Exception:
                                    out_x[idx], out_y[idx] = 0.0, 0.0
                        t1 = time.time()
                        repulsive_out[bstart:bend, 0] = out_x
                        repulsive_out[bstart:bend, 1] = out_y
                        # update last batch log entry with time
                        try:
                            if self._batch_log:
                                self._batch_log[-1]['time'] = t1 - t0
                        except Exception:
                            pass
                        # adaptive memory check moved to after batch assembly (use per-method samples)
                        pass
                    except Exception:
                        # batch-level failure: fall back to per-agent compute for this batch
                        for idx, a in enumerate(range(bstart, bend)):
                            try:
                                repulsive_out[a, :] = self._calculate_repulsive_force(mem_sections[a], a, int(row_min[a]), int(row_max[a]), int(col_min[a]), int(col_max[a]), weight, t)
                            except Exception:
                                repulsive_out[a, :] = np.array([0.0, 0.0])
                # sample rss after finishing all batches (single sample)
                try:
                    if ps_proc is not None:
                        rss_after_method = ps_proc.memory_info().rss
                except Exception:
                    rss_after_method = None

                # Update batch_log entries with method-level rss samples where available
                try:
                    if self._batch_log:
                        for entry in self._batch_log:
                            if 'rss_before' not in entry or entry.get('rss_before') is None:
                                entry['rss_before'] = rss_before_method
                            if 'rss_after' not in entry or entry.get('rss_after') is None:
                                entry['rss_after'] = rss_after_method
                except Exception:
                    pass

                # if memory spiked beyond threshold, reduce batch size for future batches based on method-level samples
                try:
                    if rss_before_method is not None and rss_after_method is not None and (rss_after_method - rss_before_method) > self._rss_threshold_bytes:
                        new_bs = max(16, int(batch_size // 2))
                        setattr(self.simulation, 'behavior_batch_size', new_bs)
                        batch_size = new_bs
                except Exception:
                    pass

                repulsive_forces_per_agent = repulsive_out
            except Exception:
                # catastrophic fallback: revert to original per-agent computation
                repulsive_forces_per_agent = np.array([
                    self._calculate_repulsive_force(mem_sections[int(agent_idx)], int(agent_idx), int(row_min[int(agent_idx)]), int(row_max[int(agent_idx)]), int(col_min[int(agent_idx)]), int(col_max[int(agent_idx)]), weight, t)
                    for agent_idx in np.arange(agent_count)
                ])

        # dump per-batch CSV log for offline analysis
        # Only write detailed per-batch CSVs when debugging or explicit SWEEP_DEBUG is set
        try:
            write_batch_logs = bool(getattr(self.simulation, 'debug_behavior', False) or os.environ.get('SWEEP_DEBUG'))
            if write_batch_logs:
                import csv
                ts = int(time.time())
                outdir = os.path.join(os.getcwd(), 'outputs', 'profiling')
                os.makedirs(outdir, exist_ok=True)
                run_tag = None
                try:
                    run_tag = getattr(self.simulation, 'run_tag', None) or os.environ.get('RUN_TAG')
                except Exception:
                    run_tag = None
                if run_tag:
                    fname = os.path.join(outdir, f'batch_log_n{agent_count}_b{int(batch_size)}_{run_tag}_{ts}.csv')
                else:
                    fname = os.path.join(outdir, f'batch_log_n{agent_count}_b{int(batch_size)}_{ts}.csv')
                with open(fname, 'w', newline='', encoding='utf-8') as fh:
                    fieldnames = ['start','end','batch_total','time','rss_before','rss_after','n_agents','batch_size','run_tag']
                    w = csv.DictWriter(fh, fieldnames=fieldnames)
                    w.writeheader()
                    for r in self._batch_log:
                        w.writerow({k: r.get(k) for k in fieldnames})
        except Exception:
            pass

        # Debug: write raw repulsive vectors only when debug_behavior is enabled
        if getattr(self.simulation, 'debug_behavior', False):
            try:
                # Prefer the simulation diagnostics writer which may manage HDF5 safely
                dw = getattr(self.simulation, 'diagnostics_writer', None)
                if dw is not None:
                    try:
                        dw.write_step(int(t), {'already_been_here': repulsive_forces_per_agent.astype('f4')})
                    except Exception:
                        # fall through to NPZ enqueue fallback
                        self._safe_npz_dump(os.path.join(os.getcwd(), 'outputs', 'diagnostics'), f'debug_already_been_here_step_{int(t)}', {'already_been_here': repulsive_forces_per_agent})
                else:
                    # HDF5 exclusive writes can block; instead enqueue as NPZ via diagnostics queue
                    self._safe_npz_dump(os.path.join(os.getcwd(), 'outputs', 'diagnostics'), f'debug_already_been_here_step_{int(t)}', {'already_been_here': repulsive_forces_per_agent})
            except Exception as ex:
                # non-fatal debug failure; swallow
                try:
                    import logging
                    logging.getLogger(__name__).debug('debug already_been_here write failed: %s', ex)
                except Exception:
                    pass

        return repulsive_forces_per_agent

    def _calculate_repulsive_force(self, mmap, agent_idx, row_min, row_max, col_min, col_max, weight, t):
        # mmap may be provided by caller for batched access; otherwise read from HDF5
        if mmap is None:
            h5 = hdf5_io.get_hdf5_obj(self.simulation)
            mmap = hdf5_io.read_dataset(h5, f'memory/{agent_idx}', default=np.zeros((1, 1)))
        mmap_section = mmap[row_min:row_max, col_min:col_max]
        t_since = t - mmap_section
        multiplier = np.where((t_since > 10) & (t_since < 7200), 1 - (t_since - 5) / (7195), 0)

        # Convert mental-map pixel indices to world coordinates (pixel centers)
        rows_idx = np.arange(row_min, row_max)
        cols_idx = np.arange(col_min, col_max)
        if rows_idx.size == 0 or cols_idx.size == 0:
            return np.array([0.0, 0.0])

        # Try to reuse a cached world grid for this window to avoid repeated meshgrid/pixel->geo calls
        cache_key = (int(row_min), int(row_max), int(col_min), int(col_max))
        if not hasattr(self, '_world_grid_cache'):
            self._world_grid_cache = {}
        try:
            world_x, world_y = self._world_grid_cache[cache_key]
        except KeyError:
            col_grid, row_grid = np.meshgrid(cols_idx, rows_idx)
            try:
                world_x, world_y = pixel_to_geo(self.simulation.mental_map_transform, row_grid, col_grid)
                # pixel_to_geo may return scalars for 1x1 windows; coerce to arrays
                if not hasattr(world_x, 'ravel'):
                    world_x = np.array([[float(world_x)]])
                if not hasattr(world_y, 'ravel'):
                    world_y = np.array([[float(world_y)]])
            except Exception:
                world_x = col_grid.astype(float)
                world_y = row_grid.astype(float)
            # store in cache (small windows only)
            try:
                self._world_grid_cache[cache_key] = (world_x, world_y)
            except Exception:
                pass

        agent_x = float(self.simulation.X[agent_idx])
        agent_y = float(self.simulation.Y[agent_idx])

        # Prepare flattened multiplier for the numeric core for best performance
        mult_flat = np.asarray(multiplier.ravel())
        wx = np.asarray(world_x.ravel())
        wy = np.asarray(world_y.ravel())
        # Use JIT-accelerated core when available, otherwise vectorized fallback
        try:
            if _NUMBA_AVAILABLE and '_repulsive_core_safe' in globals():
                tx, ty = _repulsive_core_safe(agent_x, agent_y, wx, wy, mult_flat, float(weight))
            else:
                tx, ty = _repulsive_core(agent_x, agent_y, wx, wy, mult_flat, float(weight))
            return np.array([tx, ty])
        except Exception:
            delta_x = agent_x - world_x
            delta_y = agent_y - world_y
            magnitudes = np.sqrt(delta_x**2 + delta_y**2)
            magnitudes = np.where(magnitudes == 0, 1e-6, magnitudes)

            unit_vector_x = delta_x / magnitudes
            unit_vector_y = delta_y / magnitudes

            # force scales with multiplier and inversely with distance (in world units)
            x_force = ((weight * unit_vector_x) / magnitudes) * multiplier
            y_force = ((weight * unit_vector_y) / magnitudes) * multiplier

            total_x_force = np.nansum(x_force)
            total_y_force = np.nansum(y_force)

            return np.array([total_x_force, total_y_force])

    def find_nearest_refuge(self, weight):
        """Attract agents toward the nearest refugia cell.

        Preferred data source: `environment/refugia` (shared raster, 1 indicates refuge).
        Legacy fallback: per-agent `refugia/<agent_idx>` datasets.
        """
        x = np.nan_to_num(self.simulation.X)
        y = np.nan_to_num(self.simulation.Y)
        # Prefer a shared environment refugia raster when available.
        shared_refugia = self._get_env('environment/refugia', default=None)
        if shared_refugia is not None:
            try:
                shared_refugia = np.asarray(shared_refugia)
                if shared_refugia.ndim == 2 and shared_refugia.size > 1:
                    transform = getattr(self.simulation, 'refugia_map_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)

                    # Cache a global "nearest refugia cell" index map so agents
                    # always get a direction even when no refugia are within a small window.
                    try:
                        cache = getattr(self.simulation, '_refugia_nearest_indices', None)
                    except Exception:
                        cache = None
                    need_recompute = True
                    try:
                        if cache is not None:
                            # Preferred cache format: (rows, cols) tuple
                            if isinstance(cache, tuple) and len(cache) == 2:
                                if np.asarray(cache[0]).shape == shared_refugia.shape and np.asarray(cache[1]).shape == shared_refugia.shape:
                                    need_recompute = False
                            # Back-compat: old cache stored as ndarray shaped (2, H, W)
                            elif isinstance(cache, np.ndarray) and cache.ndim == 3 and cache.shape[0] == 2 and cache.shape[1:] == shared_refugia.shape:
                                cache = (np.asarray(cache[0]), np.asarray(cache[1]))
                                try:
                                    setattr(self.simulation, '_refugia_nearest_indices', cache)
                                except Exception:
                                    pass
                                need_recompute = False
                    except Exception:
                        need_recompute = True
                    if need_recompute:
                        refuge_mask = (shared_refugia == 1)
                        if not np.any(refuge_mask):
                            return np.zeros((self.simulation.num_agents, 2), dtype=float)
                        # distance_transform_edt computes distance to nearest zero;
                        # set refugia cells to 0 by using "non-refugia" as the input mask.
                        try:
                            _, inds = distance_transform_edt(~refuge_mask, return_indices=True)
                            # `return_indices=True` returns an array shaped (ndim, H, W);
                            # store as a tuple for stable shape checks.
                            cache = (np.asarray(inds[0]), np.asarray(inds[1]))
                            try:
                                setattr(self.simulation, '_refugia_nearest_indices', cache)
                            except Exception:
                                pass
                        except Exception:
                            return np.zeros((self.simulation.num_agents, 2), dtype=float)

                    rows, cols = geo_to_pixel(x, y, transform)
                    rows = np.atleast_1d(rows).astype(int)
                    cols = np.atleast_1d(cols).astype(int)
                    rows = np.clip(rows, 0, shared_refugia.shape[0] - 1)
                    cols = np.clip(cols, 0, shared_refugia.shape[1] - 1)
                    ref_r = np.asarray(cache[0])[rows, cols]
                    ref_c = np.asarray(cache[1])[rows, cols]
                    ref_x, ref_y = pixel_to_geo(transform, ref_r, ref_c)
                    dx = np.asarray(ref_x, dtype=float) - np.asarray(self.simulation.X, dtype=float)
                    dy = np.asarray(ref_y, dtype=float) - np.asarray(self.simulation.Y, dtype=float)
                    dist = np.sqrt(dx * dx + dy * dy)
                    dist_safe = np.where(dist == 0, 1e-6, dist)
                    out = np.zeros((self.simulation.num_agents, 2), dtype=float)
                    out[:, 0] = float(weight) * dx / dist_safe
                    out[:, 1] = float(weight) * dy / dist_safe
                    out[~np.isfinite(out)] = 0.0
                    out[dist == 0] = 0.0
                    return out
            except Exception:
                # fall through to legacy method
                pass

        # Legacy per-agent refugia maps.
        try:
            transform = getattr(self.simulation, 'refugia_map_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
            refugia_map_rows, refugia_map_cols = geo_to_pixel(x, y, transform)
            buff = int(getattr(self.simulation, 'refugia_search_radius_cells', 50))
            refugia0 = hdf5_io.read_dataset(h5, 'refugia/0', default=np.zeros((1, 1)))
            row_min = np.clip(refugia_map_rows - buff, 0, None)
            row_max = np.clip(refugia_map_rows + buff + 1, None, refugia0.shape[0])
            col_min = np.clip(refugia_map_cols - buff, 0, None)
            col_max = np.clip(refugia_map_cols + buff + 1, None, refugia0.shape[1])

            attractive_forces_per_agent = np.array([
                self._calculate_attractive_force(agent_idx, int(rmin), int(rmax), int(cmin), int(cmax), float(weight))
                for agent_idx, rmin, rmax, cmin, cmax in zip(
                    np.arange(self.simulation.num_agents),
                    row_min,
                    row_max,
                    col_min,
                    col_max,
                )
            ])
            return attractive_forces_per_agent
        except Exception:
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

    def _calculate_attractive_force(self, agent_idx, row_min, row_max, col_min, col_max, weight):
        # ensure we have the hdf5-like object available (works with h5py.File or dict-like mocks)
        h5 = hdf5_io.get_hdf5_obj(self.simulation)
        refugia = hdf5_io.read_dataset(h5, f'refugia/{agent_idx}', default=np.zeros((1, 1)))
        refugia_section = refugia[row_min:row_max, col_min:col_max]
        refuge_mask = (refugia_section == 1)
        if not np.any(refuge_mask):
            return np.array([0.0, 0.0])
        try:
            # find nearest refuge to the agent (in pixel space)
            rr, cc = np.where(refuge_mask)
            if rr.size == 0:
                return np.array([0.0, 0.0])
            transform = getattr(self.simulation, 'refugia_map_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
            ar, ac = geo_to_pixel(float(self.simulation.X[agent_idx]), float(self.simulation.Y[agent_idx]), transform)
            ar = int(ar) - int(row_min)
            ac = int(ac) - int(col_min)
            d2 = (rr - ar) ** 2 + (cc - ac) ** 2
            j = int(np.argmin(d2))
            ref_r = int(rr[j]) + int(row_min)
            ref_c = int(cc[j]) + int(col_min)
            ref_x, ref_y = pixel_to_geo(transform, ref_r, ref_c)
            dx = float(ref_x) - float(self.simulation.X[agent_idx])
            dy = float(ref_y) - float(self.simulation.Y[agent_idx])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= 0.0 or not np.isfinite(dist):
                return np.array([0.0, 0.0])
            return np.array([float(weight) * dx / dist, float(weight) * dy / dist])
        except Exception:
            return np.array([0.0, 0.0])

    def vel_cue(self, weight):
        length_numpy = self.simulation.length
        buff = 2
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        rows = np.atleast_1d(rows).astype(np.int32, copy=False)
        cols = np.atleast_1d(cols).astype(np.int32, copy=False)

        vel_ds = np.asarray(self._get_env('environment/vel_mag', default=np.zeros((1, 1))), dtype=float)
        if vel_ds.ndim != 2 or vel_ds.size <= 1:
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        rr, cc, valid = self._window_rc(rows, cols, buff, vel_ds.shape)
        rr0 = np.clip(rr, 0, vel_ds.shape[0] - 1)
        cc0 = np.clip(cc, 0, vel_ds.shape[1] - 1)
        vel3d = vel_ds[rr0, cc0]
        # exclude out-of-bounds and nodata / invalid values
        vel3d = np.where(valid & np.isfinite(vel3d) & (vel3d > -9990.0), vel3d, np.inf)

        # compute dx,dy to each candidate cell (world coords) and front mask
        agent_x = np.nan_to_num(np.asarray(self.simulation.X, dtype=float)).reshape((-1,))
        agent_y = np.nan_to_num(np.asarray(self.simulation.Y, dtype=float)).reshape((-1,))
        heading = np.asarray(self.simulation.heading, dtype=float).reshape((-1,))
        transform = getattr(self.simulation, 'vel_mag_rast_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
        dx, dy = self._window_dxdy(rr0, cc0, transform, agent_x, agent_y)
        hx = np.cos(heading)[:, np.newaxis, np.newaxis]
        hy = np.sin(heading)[:, np.newaxis, np.newaxis]
        front = (dx * hx + dy * hy) > 0
        front_any = np.any(front & valid, axis=(1, 2))
        vel3d = np.where(front_any[:, np.newaxis, np.newaxis], np.where(front, vel3d, np.inf), vel3d)

        # If there are no finite candidates, return zero vector.
        best_val = np.min(vel3d, axis=(1, 2))
        ok = np.isfinite(best_val)
        if not np.any(ok):
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        flat = vel3d.reshape((vel3d.shape[0], -1))
        best_idx = np.argmin(flat, axis=1)
        H = 2 * buff + 1
        rr_i = best_idx // H
        cc_i = best_idx % H

        sel_dx = dx[np.arange(dx.shape[0]), rr_i, cc_i]
        sel_dy = dy[np.arange(dy.shape[0]), rr_i, cc_i]
        dist = np.sqrt(sel_dx * sel_dx + sel_dy * sel_dy)
        dist_safe = np.where(dist == 0, 1e-6, dist)
        attract_x = float(weight) * sel_dx / dist_safe
        attract_y = float(weight) * sel_dy / dist_safe
        attract_x = np.where(ok & (dist != 0), attract_x, 0.0)
        attract_y = np.where(ok & (dist != 0), attract_y, 0.0)
        return np.column_stack((attract_x, attract_y))

    def rheo_cue(self, weight, downstream=False):
        sampler = getattr(self.simulation, 'sample_environment', None)
        if not callable(sampler):
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        sign = -1.0 if not downstream else 1.0
        tx = getattr(self.simulation, 'vel_x_rast_transform', None) or getattr(self.simulation, 'vel_dir_rast_transform', None)
        ty = getattr(self.simulation, 'vel_y_rast_transform', None) or getattr(self.simulation, 'vel_dir_rast_transform', None)

        x_vel = sign * np.asarray(sampler(tx, 'vel_x'), dtype=float)
        y_vel = sign * np.asarray(sampler(ty, 'vel_y'), dtype=float)

        # fallback to vel_dir transform if primary transforms produce all-NaN values
        if not (np.isfinite(x_vel).any() or np.isfinite(y_vel).any()):
            tdir = getattr(self.simulation, 'vel_dir_rast_transform', None)
            if tdir is not None:
                x_vel = sign * np.asarray(sampler(tdir, 'vel_x'), dtype=float)
                y_vel = sign * np.asarray(sampler(tdir, 'vel_y'), dtype=float)

        v = np.column_stack([x_vel, y_vel])
        # store sampled velocities for debugging/inspection by NPZ dumps
        # ensure array shape (n_agents,2) — best-effort
        self._safe_set_sim_attr('last_sampled_vel', self._safe_asarray(v, dtype=float, default=None))
        # sanitize sampled values (handle nodata values like -9999 and zeros)
        v = np.asarray(v, dtype=float)
        mags = np.linalg.norm(v, axis=-1)
        # treat nodata / enormous values as zero (no rheotaxis)
        invalid = ~np.isfinite(mags) | (mags <= 0) | (mags > 1e6)
        v_hat = np.zeros_like(v)
        valid = ~invalid
        if np.any(valid):
            v_hat[valid] = (v[valid].T / mags[valid]).T
        rheotaxis = weight * v_hat
        return rheotaxis

    def border_cue(self, weight, t):
        length_numpy = self.simulation.length
        buff = 2
        x, y = (np.nan_to_num(self.simulation.X), np.nan_to_num(self.simulation.Y))
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        rows = np.atleast_1d(rows).astype(np.int32, copy=False)
        cols = np.atleast_1d(cols).astype(np.int32, copy=False)

        dist_ds = np.asarray(self._get_env('environment/distance_to', default=np.zeros((1, 1))), dtype=float)
        if dist_ds.ndim != 2 or dist_ds.size <= 1:
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        rr, cc, valid = self._window_rc(rows, cols, buff, dist_ds.shape)
        rr0 = np.clip(rr, 0, dist_ds.shape[0] - 1)
        cc0 = np.clip(cc, 0, dist_ds.shape[1] - 1)
        dist3d = dist_ds[rr0, cc0]
        dist3d = np.where(valid & np.isfinite(dist3d), dist3d, -np.inf)

        agent_x = np.nan_to_num(np.asarray(self.simulation.X, dtype=float)).reshape((-1,))
        agent_y = np.nan_to_num(np.asarray(self.simulation.Y, dtype=float)).reshape((-1,))
        heading = np.asarray(self.simulation.heading, dtype=float).reshape((-1,))
        transform = getattr(self.simulation, 'vel_mag_rast_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
        dx, dy = self._window_dxdy(rr0, cc0, transform, agent_x, agent_y)
        hx = np.cos(heading)[:, np.newaxis, np.newaxis]
        hy = np.sin(heading)[:, np.newaxis, np.newaxis]
        front = (dx * hx + dy * hy) > 0
        front_any = np.any(front & valid, axis=(1, 2))
        dist3d = np.where(front_any[:, np.newaxis, np.newaxis], np.where(front, dist3d, -np.inf), dist3d)

        flat = dist3d.reshape((dist3d.shape[0], -1))
        best_idx = np.argmax(flat, axis=1)
        H = 2 * buff + 1
        rr_i = best_idx // H
        cc_i = best_idx % H

        sel_dx = dx[np.arange(dx.shape[0]), rr_i, cc_i]
        sel_dy = dy[np.arange(dy.shape[0]), rr_i, cc_i]
        dist = np.sqrt(sel_dx * sel_dx + sel_dy * sel_dy)

        current_distances = self.simulation.sample_environment(self.simulation.depth_rast_transform, 'distance_to')
        self.simulation.current_distances = current_distances

        # Scale repulsion by distance to boundary so the cue remains meaningful on
        # coarser rasters (e.g., 1m cells) while preserving legacy behavior when
        # fish are truly near the edge.
        try:
            pw = abs(float(_unpack_affine(self.simulation.depth_rast_transform)[0]))
        except Exception:
            pw = 0.0
        length_m = np.asarray(self.simulation.length, dtype=float) / 1000.0
        base_influence = getattr(self.simulation, 'border_influence_distance_m', None)
        if base_influence is None:
            influence_dist = np.maximum(10.0, np.maximum(10.0 * length_m, pw))
        else:
            try:
                influence_dist = np.full(self.simulation.num_agents, float(base_influence), dtype=float)
            except Exception:
                influence_dist = np.asarray(base_influence, dtype=float)
        influence_dist = np.where(influence_dist <= 0, 10.0, influence_dist)
        influence = (influence_dist - current_distances) / influence_dist
        influence = np.clip(influence, 0.0, 1.0)
        influence = np.where(~np.isfinite(influence), 0.0, influence)
        influence = np.where(self.simulation.in_eddy == 1, 1.0, influence)

        dist_safe = np.where(dist == 0, 1e-6, dist)
        repulse_x = float(weight) * influence * sel_dx / dist_safe
        repulse_y = float(weight) * influence * sel_dy / dist_safe

        return np.column_stack((repulse_x, repulse_y))

    def shallow_cue(self, weight):
        buff = 2
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        rows = np.atleast_1d(rows).astype(np.int32, copy=False)
        cols = np.atleast_1d(cols).astype(np.int32, copy=False)

        depth_ds = np.asarray(self._get_env('environment/depth', default=np.zeros((1, 1))), dtype=float)
        if depth_ds.ndim != 2 or depth_ds.size <= 1:
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        rr, cc, valid = self._window_rc(rows, cols, buff, depth_ds.shape)
        rr0 = np.clip(rr, 0, depth_ds.shape[0] - 1)
        cc0 = np.clip(cc, 0, depth_ds.shape[1] - 1)
        depths = depth_ds[rr0, cc0]

        agent_x = np.nan_to_num(np.asarray(self.simulation.X, dtype=float)).reshape((-1,))
        agent_y = np.nan_to_num(np.asarray(self.simulation.Y, dtype=float)).reshape((-1,))
        heading = np.asarray(self.simulation.heading, dtype=float).reshape((-1,))
        transform = getattr(self.simulation, 'vel_mag_rast_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
        dx_cell, dy_cell = self._window_dxdy(rr0, cc0, transform, agent_x, agent_y)
        # for repulsion, use vector from cell->agent
        dx = -dx_cell
        dy = -dy_cell

        hx = np.cos(heading)[:, np.newaxis, np.newaxis]
        hy = np.sin(heading)[:, np.newaxis, np.newaxis]
        front = (dx_cell * hx + dy_cell * hy) > 0
        front_any = np.any(front & valid, axis=(1, 2))
        front_mult = np.where(front_any[:, np.newaxis, np.newaxis], front, True).astype(np.float64)

        min_depth = np.asarray(self.simulation.too_shallow, dtype=float).reshape((-1,))
        depth_mult = (depths < min_depth[:, np.newaxis, np.newaxis]).astype(np.float64)
        depth_mult *= valid.astype(np.float64)

        dist2 = dx * dx + dy * dy
        dist2 = np.where(dist2 == 0, 1e-6, dist2)
        x_force = (float(weight) * dx / dist2) * depth_mult * front_mult
        y_force = (float(weight) * dy / dist2) * depth_mult * front_mult
        total_x_force = np.nansum(x_force, axis=(1, 2))
        total_y_force = np.nansum(y_force, axis=(1, 2))
        return np.column_stack((total_x_force, total_y_force))

    def wave_drag_multiplier(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/wave_drag_huges_2004_fig3.csv')
        hughes = pd.read_csv(data_dir)
        hughes.sort_values(by='body_depths_submerged', ascending=True, inplace=True)
        wave_drag_fun = UnivariateSpline(hughes.body_depths_submerged, hughes.wave_drag_multiplier, k=3, ext=0)
        body_depths = self.simulation.z / (self.simulation.body_depth / 100.)
        self.simulation.wave_drag = np.where(body_depths >= 3, 1, wave_drag_fun(body_depths))

    def wave_drag_cue(self, weight):
        buff = 2
        x, y = (self.simulation.X, self.simulation.Y)
        rows, cols = geo_to_pixel(x, y, self.simulation.depth_rast_transform)
        rows = np.atleast_1d(rows).astype(np.int32, copy=False)
        cols = np.atleast_1d(cols).astype(np.int32, copy=False)

        depth_ds = np.asarray(self._get_env('environment/depth', default=np.zeros((1, 1))), dtype=float)
        if depth_ds.ndim != 2 or depth_ds.size <= 1:
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        rr, cc, valid = self._window_rc(rows, cols, buff, depth_ds.shape)
        rr0 = np.clip(rr, 0, depth_ds.shape[0] - 1)
        cc0 = np.clip(cc, 0, depth_ds.shape[1] - 1)
        dep3d = depth_ds[rr0, cc0]
        dep3d = np.where(valid & np.isfinite(dep3d) & (dep3d > -9990.0), dep3d, np.inf)

        agent_x = np.nan_to_num(np.asarray(self.simulation.X, dtype=float)).reshape((-1,))
        agent_y = np.nan_to_num(np.asarray(self.simulation.Y, dtype=float)).reshape((-1,))
        heading = np.asarray(self.simulation.heading, dtype=float).reshape((-1,))
        transform = getattr(self.simulation, 'vel_mag_rast_transform', None) or getattr(self.simulation, 'depth_rast_transform', None)
        dx, dy = self._window_dxdy(rr0, cc0, transform, agent_x, agent_y)
        hx = np.cos(heading)[:, np.newaxis, np.newaxis]
        hy = np.sin(heading)[:, np.newaxis, np.newaxis]
        front = (dx * hx + dy * hy) > 0
        front_any = np.any(front & valid, axis=(1, 2))
        dep3d = np.where(front_any[:, np.newaxis, np.newaxis], np.where(front, dep3d, np.inf), dep3d)

        opt = np.asarray(self.simulation.opt_wat_depth, dtype=float).reshape((-1, 1, 1))
        score = np.abs(dep3d - opt)
        best_val = np.min(score, axis=(1, 2))
        ok = np.isfinite(best_val)
        if not np.any(ok):
            return np.zeros((self.simulation.num_agents, 2), dtype=float)

        flat = score.reshape((score.shape[0], -1))
        best_idx = np.argmin(flat, axis=1)
        H = 2 * buff + 1
        rr_i = best_idx // H
        cc_i = best_idx % H
        sel_dx = dx[np.arange(dx.shape[0]), rr_i, cc_i]
        sel_dy = dy[np.arange(dy.shape[0]), rr_i, cc_i]
        dist = np.sqrt(sel_dx * sel_dx + sel_dy * sel_dy)
        dist_safe = np.where(dist == 0, 1e-6, dist)
        attract_x = float(weight) * sel_dx / dist_safe
        attract_y = float(weight) * sel_dy / dist_safe
        attract_x = np.where(ok & (dist != 0), attract_x, 0.0)
        attract_y = np.where(ok & (dist != 0), attract_y, 0.0)
        return np.column_stack((attract_x, attract_y))

    def cohesion_cue(self, weight, consider_front_only=False):
        num_agents = int(self.simulation.num_agents)
        # Prefer CSR neighbor representation when available (simulation-level)
        offsets = getattr(self.simulation, 'neighbors_offsets', None)
        neighbors = getattr(self.simulation, 'neighbors_indices', None)
        if offsets is not None and neighbors is not None:
            offsets = np.asarray(offsets, dtype=np.int64).reshape((-1,))
            neighbors = np.asarray(neighbors, dtype=np.int32).reshape((-1,))
            if offsets.size != num_agents + 1 or neighbors.size < int(offsets[-1] if offsets.size else 0):
                offsets = None
                neighbors = None

        if offsets is None or neighbors is None:
            awb = getattr(self.simulation, 'agents_within_buffers', None) or []
            if len(awb) != num_agents:
                return np.zeros((num_agents, 2), dtype=float)
            lengths = np.fromiter((len(nbrs) for nbrs in awb), count=num_agents, dtype=np.int32)
            offsets = np.empty(num_agents + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            total = int(offsets[-1])
            if total <= 0:
                return np.zeros((num_agents, 2), dtype=float)
            neighbors = np.concatenate(awb).astype(np.int32, copy=False)

        # Neighbor lengths per agent (used by fallback and for sanity checks)
        try:
            lengths = np.diff(np.asarray(offsets, dtype=np.int64)).astype(np.int32, copy=False)
        except Exception:
            lengths = np.zeros(num_agents, dtype=np.int32)

        X = np.asarray(self.simulation.X, dtype=np.float64).reshape((-1,))
        Y = np.asarray(self.simulation.Y, dtype=np.float64).reshape((-1,))
        x_vel = np.asarray(getattr(self.simulation, 'x_vel', np.zeros(num_agents)), dtype=np.float64).reshape((-1,))
        y_vel = np.asarray(getattr(self.simulation, 'y_vel', np.zeros(num_agents)), dtype=np.float64).reshape((-1,))

        if _NUMBA_AVAILABLE:
            outx = np.zeros(num_agents, dtype=np.float64)
            outy = np.zeros(num_agents, dtype=np.float64)
            _cohesion_from_neighbors_offsets(
                np.ascontiguousarray(offsets, dtype=np.int64),
                np.ascontiguousarray(neighbors, dtype=np.int32),
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(Y, dtype=np.float64),
                np.ascontiguousarray(x_vel, dtype=np.float64),
                np.ascontiguousarray(y_vel, dtype=np.float64),
                np.int8(1 if consider_front_only else 0),
                outx,
                outy,
            )
            out = np.column_stack((outx, outy))
            out *= float(weight)
            out[~np.isfinite(out)] = 0.0
            return out

        # Fallback: keep existing vectorized approach when Numba isn't available.
        neighbor_indices = np.asarray(neighbors, dtype=np.int32).reshape((-1,))
        agent_indices = np.repeat(np.arange(num_agents), lengths).astype(np.int32, copy=False)
        valid_nb = (neighbor_indices >= 0) & (neighbor_indices != agent_indices)
        if np.any(~valid_nb):
            neighbor_indices = neighbor_indices[valid_nb]
            agent_indices = agent_indices[valid_nb]
        x_neighbors = X[neighbor_indices]
        y_neighbors = Y[neighbor_indices]
        vectors_to_neighbors_x = x_neighbors - X[agent_indices]
        vectors_to_neighbors_y = y_neighbors - Y[agent_indices]

        if consider_front_only:
            agent_velocities_x = x_vel[agent_indices]
            agent_velocities_y = y_vel[agent_indices]
            dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
            valid_neighbors_mask = dot_products > 0
        else:
            valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)

        valid_neighbor_indices = neighbor_indices[valid_neighbors_mask]
        valid_agent_indices = agent_indices[valid_neighbors_mask]

        center_x = np.zeros(num_agents)
        center_y = np.zeros(num_agents)
        np.add.at(center_x, valid_agent_indices, x_neighbors[valid_neighbors_mask])
        np.add.at(center_y, valid_agent_indices, y_neighbors[valid_neighbors_mask])
        counts = np.bincount(valid_agent_indices, minlength=num_agents)
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        center_x = center_x / counts_safe
        center_y = center_y / counts_safe
        no_neighbors = counts == 0
        if np.any(no_neighbors):
            center_x[no_neighbors] = X[no_neighbors]
            center_y[no_neighbors] = Y[no_neighbors]

        vectors_to_center_x = center_x - X
        vectors_to_center_y = center_y - Y
        distances_to_center = np.sqrt(vectors_to_center_x**2 + vectors_to_center_y**2)
        epsilon = 1e-10
        v_hat_center_x = np.divide(vectors_to_center_x, distances_to_center + epsilon, out=np.zeros_like(x_vel), where=distances_to_center + epsilon != 0)
        v_hat_center_y = np.divide(vectors_to_center_y, distances_to_center + epsilon, out=np.zeros_like(y_vel), where=distances_to_center + epsilon != 0)
        cohesion_array = np.zeros((num_agents, 2))
        cohesion_array[:, 0] = float(weight) * v_hat_center_x
        cohesion_array[:, 1] = float(weight) * v_hat_center_y
        return np.nan_to_num(cohesion_array)

    def alignment_cue(self, weight, consider_front_only=False):
        num_agents = int(self.simulation.num_agents)
        # Prefer CSR neighbor representation when available (simulation-level)
        offsets = getattr(self.simulation, 'neighbors_offsets', None)
        neighbor_indices = getattr(self.simulation, 'neighbors_indices', None)
        if offsets is not None and neighbor_indices is not None:
            offsets = np.asarray(offsets, dtype=np.int64).reshape((-1,))
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32).reshape((-1,))
            total = int(offsets[-1]) if offsets.size else 0
            if offsets.size != num_agents + 1 or neighbor_indices.size < total:
                offsets = None
                neighbor_indices = None
        if offsets is None or neighbor_indices is None:
            awb = getattr(self.simulation, 'agents_within_buffers', None) or []
            if len(awb) != num_agents:
                return np.zeros((num_agents, 2), dtype=float)
            lengths = np.fromiter((len(nbrs) for nbrs in awb), count=num_agents, dtype=np.int32)
            offsets = np.empty(num_agents + 1, dtype=np.int64)
            offsets[0] = 0
            offsets[1:] = np.cumsum(lengths, dtype=np.int64)
            total = int(offsets[-1])
            if total <= 0:
                self.simulation.school_sog = np.maximum(0.5 * (np.asarray(self.simulation.length, dtype=float) / 1000.0), 0.0)
                return np.zeros((num_agents, 2), dtype=float)
            neighbor_indices = np.concatenate(awb).astype(np.int32, copy=False)
        else:
            total = int(offsets[-1]) if offsets.size else 0
            if total <= 0:
                self.simulation.school_sog = np.maximum(0.5 * (np.asarray(self.simulation.length, dtype=float) / 1000.0), 0.0)
                return np.zeros((num_agents, 2), dtype=float)
        # Neighbor lengths per agent (needed for fallback and for diagnostic sanity)
        try:
            lengths = np.diff(np.asarray(offsets, dtype=np.int64)).astype(np.int32, copy=False)
        except Exception:
            lengths = np.zeros(num_agents, dtype=np.int32)
        # capture raw neighbor headings (may be all zeros at init)
        if getattr(self.simulation, 'debug_behavior', False):
            try:
                import logging
                logging.getLogger(__name__).debug('alignment_cue ENTER: num_agents=%s', num_agents)
                logging.getLogger(__name__).debug('agents_within_buffers lengths=%s', [len(x) for x in self.simulation.agents_within_buffers])
                logging.getLogger(__name__).debug('neighbor_indices sample=%s', neighbor_indices[:20])
                logging.getLogger(__name__).debug('sim.heading sample=%s', np.asarray(self.simulation.heading)[:20])
            except Exception:
                # non-fatal: continue without verbose logs
                pass
        # read raw headings; if unavailable, use empty array
        raw_headings_neighbors = np.array([], dtype=float)
        try:
            raw_headings_neighbors = np.asarray(self.simulation.heading, dtype=float)[neighbor_indices]
        except Exception:
            raw_headings_neighbors = np.array([], dtype=float)
        headings_neighbors = raw_headings_neighbors.copy()
        # If headings are all zero (common at initialization), fall back to neighbor velocity directions
        used_velocity_heading = False
        # Avoid `np.allclose` here: it is expensive and shows up as a hotspot in
        # sim profiling. We only need the common init case where headings are
        # exactly all zeros.
        if headings_neighbors.size > 0 and (not np.any(headings_neighbors)):
            # compute neighbor velocities' headings where available
            try:
                vx = np.asarray(self.simulation.x_vel)[neighbor_indices]
                vy = np.asarray(self.simulation.y_vel)[neighbor_indices]
                vel_mag = np.sqrt(vx**2 + vy**2)
                if np.any(vel_mag > 0):
                    headings_neighbors = np.arctan2(vy, vx)
                    used_velocity_heading = True
                    if getattr(self.simulation, 'debug_behavior', False):
                        try:
                            import logging
                            logging.getLogger(__name__).debug('alignment_cue: used velocity fallback; sample headings_neighbors=%s', headings_neighbors[:20])
                        except Exception:
                            pass
            except Exception:
                # do not propagate; leave headings_neighbors as-is
                pass
        # Only build heavy diagnostics when explicitly requested.
        want_diag = bool(getattr(self.simulation, 'debug_behavior', False)) or (os.environ.get('FORCE_RAWVECS', '').lower() == 'true')
        if want_diag:
            ad = {
                'raw_headings_neighbors': self._safe_asarray(raw_headings_neighbors, dtype=float, default=np.array([])),
                'headings_neighbors_used': self._safe_asarray(headings_neighbors, dtype=float, default=np.array([])),
                'used_velocity_heading': bool(used_velocity_heading),
                'neighbor_indices': self._safe_asarray(neighbor_indices, dtype=np.int32, default=np.array([], dtype=np.int32)),
                'agent_indices': np.array([], dtype=np.int32),
            }
            self._safe_set_sim_attr('_alignment_diag', ad)

        # Fast path: Numba kernel over per-agent neighbor slices.
        X = np.asarray(self.simulation.X, dtype=np.float64).reshape((-1,))
        Y = np.asarray(self.simulation.Y, dtype=np.float64).reshape((-1,))
        x_vel = np.asarray(getattr(self.simulation, 'x_vel', np.zeros(num_agents)), dtype=np.float64).reshape((-1,))
        y_vel = np.asarray(getattr(self.simulation, 'y_vel', np.zeros(num_agents)), dtype=np.float64).reshape((-1,))
        sog = np.asarray(getattr(self.simulation, 'sog', np.zeros(num_agents)), dtype=np.float64).reshape((-1,))

        if _NUMBA_AVAILABLE:
            unit_x = np.zeros(num_agents, dtype=np.float64)
            unit_y = np.zeros(num_agents, dtype=np.float64)
            mean_sog = np.full(num_agents, np.nan, dtype=np.float64)
            counts = np.zeros(num_agents, dtype=np.int32)
            _alignment_from_neighbors_offsets(
                np.ascontiguousarray(offsets, dtype=np.int64),
                np.ascontiguousarray(neighbor_indices, dtype=np.int32),
                np.ascontiguousarray(np.asarray(headings_neighbors, dtype=np.float64), dtype=np.float64),
                np.ascontiguousarray(X, dtype=np.float64),
                np.ascontiguousarray(Y, dtype=np.float64),
                np.ascontiguousarray(x_vel, dtype=np.float64),
                np.ascontiguousarray(y_vel, dtype=np.float64),
                np.ascontiguousarray(sog, dtype=np.float64),
                np.int8(1 if consider_front_only else 0),
                unit_x,
                unit_y,
                mean_sog,
                counts,
            )
            alignment_array = np.column_stack((unit_x, unit_y)) * float(weight)
            alignment_array[~np.isfinite(alignment_array)] = 0.0
        else:
            # Fallback: previous vectorized implementation.
            agent_indices = np.repeat(np.arange(num_agents), lengths).astype(np.int32, copy=False)
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32).reshape((-1,))
            valid_nb = (neighbor_indices >= 0) & (neighbor_indices != agent_indices)
            if np.any(~valid_nb):
                neighbor_indices = neighbor_indices[valid_nb]
                agent_indices = agent_indices[valid_nb]
            vectors_to_neighbors_x = X[neighbor_indices] - X[agent_indices]
            vectors_to_neighbors_y = Y[neighbor_indices] - Y[agent_indices]
            if consider_front_only:
                agent_velocities_x = x_vel[agent_indices]
                agent_velocities_y = y_vel[agent_indices]
                dot_products = vectors_to_neighbors_x * agent_velocities_x + vectors_to_neighbors_y * agent_velocities_y
                valid_neighbors_mask = dot_products > 0
            else:
                valid_neighbors_mask = np.ones_like(neighbor_indices, dtype=bool)
            valid_agent_indices = agent_indices[valid_neighbors_mask]
            sum_cos = np.zeros(num_agents)
            sum_sin = np.zeros(num_agents)
            np.add.at(sum_cos, valid_agent_indices, np.cos(headings_neighbors[valid_neighbors_mask]))
            np.add.at(sum_sin, valid_agent_indices, np.sin(headings_neighbors[valid_neighbors_mask]))
            counts = np.bincount(valid_agent_indices, minlength=num_agents)
            counts_safe = counts.copy()
            counts_safe[counts_safe == 0] = 1
            mean_cos = sum_cos / counts_safe
            mean_sin = sum_sin / counts_safe
            avg_mag = np.sqrt(mean_cos**2 + mean_sin**2)
            avg_mag_safe = np.where(avg_mag == 0, 1e-6, avg_mag)
            v_hat_align_x = mean_cos / avg_mag_safe
            v_hat_align_y = mean_sin / avg_mag_safe
            no_school = np.where(counts == 0, 0.0, 1.0)
            alignment_array = np.zeros((num_agents, 2))
            alignment_array[:, 0] = float(weight) * v_hat_align_x * no_school
            alignment_array[:, 1] = float(weight) * v_hat_align_y * no_school
            # mean neighbor sog
            sum_sog = np.zeros(num_agents, dtype=float)
            np.add.at(sum_sog, valid_agent_indices, sog[neighbor_indices[valid_neighbors_mask]])
            mean_sog = sum_sog / counts_safe

        try:
            min_sog = 0.5 * (np.asarray(self.simulation.length, dtype=float) / 1000.0)
        except Exception:
            min_sog = 0.0
        mean_sog = np.where(np.isfinite(mean_sog), mean_sog, min_sog)
        mean_sog = np.where(counts == 0, min_sog, mean_sog)
        self.simulation.school_sog = np.maximum(mean_sog, min_sog)
        # record whether alignment used velocity-derived headings for diagnostics
        if want_diag:
            self._safe_set_sim_attr('alignment_used_velocity', bool(used_velocity_heading))
        return np.nan_to_num(alignment_array)

    def collision_cue(self, weight):
        # ensure closest_agent and nearest_neighbor_distance are populated; reconstruct when missing
        try:
            closest_agent_arr = np.asarray(self.simulation.closest_agent, dtype=float).copy()
        except Exception:
            closest_agent_arr = np.full(self.simulation.num_agents, np.nan)
        try:
            nearest_d_arr = np.asarray(self.simulation.nearest_neighbor_distance, dtype=float).copy()
        except Exception:
            nearest_d_arr = np.full(self.simulation.num_agents, np.nan)

        # reconstruct missing entries from agents_within_buffers
        try:
            awb = getattr(self.simulation, 'agents_within_buffers', None)
            if awb is not None:
                for ag in range(self.simulation.num_agents):
                    if np.isnan(nearest_d_arr[ag]) or np.isnan(closest_agent_arr[ag]):
                        nbrs = awb[ag]
                        if nbrs is None or len(nbrs) == 0:
                            continue
                        # compute distances to neighbors
                        dx = self.simulation.X[nbrs] - self.simulation.X[ag]
                        dy = self.simulation.Y[nbrs] - self.simulation.Y[ag]
                        dists = np.sqrt(dx**2 + dy**2)
                        idx = int(np.argmin(dists))
                        closest_agent_arr[ag] = nbrs[idx]
                        nearest_d_arr[ag] = float(dists[idx])
        except Exception:
            pass

        # update simulation attributes so other code sees reconstructed values
        try:
            self.simulation.closest_agent = closest_agent_arr
            self.simulation.nearest_neighbor_distance = nearest_d_arr
        except Exception:
            pass

        valid_indices = ~np.isnan(closest_agent_arr)
        closest_X = np.full_like(self.simulation.X, np.nan)
        closest_Y = np.full_like(self.simulation.Y, np.nan)
        try:
            closest_X[valid_indices] = self.simulation.X[closest_agent_arr[valid_indices].astype(int)]
            closest_Y[valid_indices] = self.simulation.Y[closest_agent_arr[valid_indices].astype(int)]
        except Exception:
            # fallback: leave NaNs
            pass

        self_2_closest = np.column_stack((closest_X.flatten() - self.simulation.X.flatten(), closest_Y.flatten() - self.simulation.Y.flatten()))
        closest_2_self = np.column_stack((self.simulation.X.flatten() - closest_X.flatten(), self.simulation.Y.flatten() - closest_Y.flatten()))

        invalid_vectors = np.isnan(closest_2_self).any(axis=1)
        closest_2_self[invalid_vectors] = [np.nan, np.nan]
        closest_2_self = np.nan_to_num(closest_2_self)

        safe_distances = np.where(self.simulation.nearest_neighbor_distance > 0, self.simulation.nearest_neighbor_distance, np.nan)
        v_hat_x = np.divide(closest_2_self[:, 0], safe_distances, out=np.zeros_like(closest_2_self[:, 0]), where=safe_distances != 0)
        v_hat_y = np.divide(closest_2_self[:, 1], safe_distances, out=np.zeros_like(closest_2_self[:, 1]), where=safe_distances != 0)

        collision_cue_x = np.divide(weight * v_hat_x, safe_distances**2, out=np.zeros_like(v_hat_x), where=safe_distances != 0)
        collision_cue_y = np.divide(weight * v_hat_y, safe_distances**2, out=np.zeros_like(v_hat_y), where=safe_distances != 0)

        collision_cue_mm = np.column_stack((collision_cue_x, collision_cue_y))
        np.nan_to_num(collision_cue_mm, copy=False)
        return collision_cue_mm

    def is_in_eddy(self, t):
        linear_positions = self.simulation.compute_linear_positions(self.simulation.longitudinal)
        self.current_longitudes = linear_positions
        self.simulation.past_longitudes[:, :-1] = self.simulation.past_longitudes[:, 1:]
        self.simulation.swim_speeds[:, :-1] = self.simulation.swim_speeds[:, 1:]
        self.simulation.past_longitudes[:, -1] = linear_positions
        self.simulation.swim_speeds[:, -1] = self.simulation.sog
        valid_entries = ~np.isnan(self.simulation.swim_speeds[:, 0]) & ~np.isnan(self.simulation.swim_speeds[:, -1])

        avg_speeds = np.full(self.simulation.swim_speeds.shape[0], np.nan)
        avg_speeds[valid_entries] = np.max(self.simulation.swim_speeds[valid_entries], axis=-1)

        total_displacement = np.full(self.simulation.past_longitudes.shape[0], np.nan)
        total_displacement[valid_entries] = self.simulation.past_longitudes[valid_entries, -1] - self.simulation.past_longitudes[valid_entries, 0]

        delta = self.simulation.past_longitudes[valid_entries, 0] - self.simulation.past_longitudes[valid_entries, -1]
        dt = self.simulation.past_longitudes.shape[1]
        expected_displacement = avg_speeds * dt
        long_dir = self.simulation.past_longitudes[:, -2] - self.simulation.past_longitudes[:, -1]

        if delta.shape == total_displacement.shape and t >= 1800.:
            stuck_conditions = (expected_displacement >= 5. * np.abs(total_displacement)) & (self.simulation.swim_behav == 1)
        else:
            stuck_conditions = np.zeros_like(self.simulation.X)

        not_in_eddy_anymore = self.simulation.time_since_eddy_escape >= self.simulation.max_eddy_escape_seconds
        self.simulation.swim_speeds[not_in_eddy_anymore, :] = np.nan
        self.simulation.past_longitudes[not_in_eddy_anymore, :] = np.nan
        self.simulation.time_since_eddy_escape[not_in_eddy_anymore] = 0.0

        already_in_eddy = self.simulation.in_eddy == True
        self.simulation.in_eddy = np.where(np.logical_or(stuck_conditions, already_in_eddy), True, False)
        self.simulation.in_eddy[not_in_eddy_anymore] = False
        self.simulation.time_since_eddy_escape[self.simulation.in_eddy == True] += 1

    def arbitrate(self, t):
        # debug: log a concise summary of current headings at start of arbitration
        if getattr(self.simulation, 'debug_behavior', False):
            try:
                h = np.asarray(self.simulation.heading)
                # show size, mean, and a short sample (first 10 entries) instead of whole array
                sample = list(h[:10]) if getattr(h, 'size', 0) > 0 else []
                mean = float(np.nanmean(h)) if getattr(h, 'size', 0) > 0 else float('nan')
                logging.getLogger(__name__).debug(
                    "arbitrate: heading size=%s mean=%s sample=%s",
                    getattr(h, 'size', 0),
                    mean,
                    sample,
                )
            except Exception:
                pass
        self._safe_set_sim_attr('heading_in', self._safe_asarray(self.simulation.heading, dtype=np.float32, default=None))
        if self.simulation.pid_tuning:
            # allow test-time override of weights via simulation.test_weights dict
            tw = getattr(self.simulation, 'test_weights', None)
            if tw and 'rheotaxis' in tw:
                rheotaxis = self.rheo_cue(float(tw.get('rheotaxis', 50000)))
            else:
                rheotaxis = self.rheo_cue(50000)
        else:
            tw = getattr(self.simulation, 'test_weights', None)
            # default weights
            # If a test_weights dict is present we treat it as authoritative:
            # start with zero weights for all known cues then apply overrides
            # so that missing keys remain zero (useful for isolated-cue tests).
            known_keys = ['rheotaxis', 'alignment', 'cohesion', 'low_speed', 'wave_drag', 'refugia', 'border', 'shallow', 'avoid', 'collision']
            if tw:
                default_weights = {k: 0.0 for k in known_keys}
                for k, v in tw.items():
                    try:
                        if k in default_weights:
                            default_weights[k] = float(v)
                    except Exception:
                        pass
            else:
                default_weights = {
                    'rheotaxis': 25000,
                    'alignment': 20500,
                    'cohesion': 11000,
                    'low_speed': 1500,
                    'wave_drag': 0,
                    'refugia': 50000,
                    'border': 50000,
                    'shallow': 100000,
                    'avoid': 25000,
                    'collision': 50000,
                }

            # ensure rheotaxis is always computed (used downstream)
            if getattr(self.simulation, 'debug_behavior', False):
                logging.getLogger(__name__).debug('DBG arbitrate: about to call alignment_cue')
            rheotaxis = self.rheo_cue(default_weights.get('rheotaxis', 25000))
            alignment = self.alignment_cue(default_weights['alignment'])
            cohesion = self.cohesion_cue(default_weights['cohesion'])
            low_speed = self.vel_cue(default_weights['low_speed'])
            wave_drag = self.wave_drag_cue(default_weights['wave_drag'])
            refugia = self.find_nearest_refuge(default_weights['refugia'])
            border = self.border_cue(default_weights['border'], t)
            shallow = self.shallow_cue(default_weights['shallow'])
            avoid = self.already_been_here(default_weights['avoid'], t)
            collision = self.collision_cue(default_weights['collision'])

        # cue application / logging order (migratory mode)
        order_dict = {
            0: 'shallow',
            1: 'border',
            2: 'avoid',
            3: 'collision',
            4: 'alignment',
            5: 'cohesion',
            6: 'low_speed',
            7: 'refugia',
            8: 'rheotaxis',
            9: 'wave_drag',
        }

        cue_dict = {'rheotaxis': rheotaxis,
                'shallow': shallow,
                'border': border,
                'wave_drag': wave_drag,
                'low_speed': low_speed,
                'avoid': avoid,
                'alignment': alignment,
                'cohesion': cohesion,
                'collision': collision,
                'refugia': refugia}

        debug_behavior = bool(getattr(self.simulation, 'debug_behavior', False))
        force_rawvecs = os.environ.get('FORCE_RAWVECS', '').lower() == 'true'
        record_state = bool(getattr(self.simulation, 'record_behavior_state', False))
        want_record = debug_behavior or force_rawvecs or record_state

        if want_record:
            self._record_cue_shapes(cue_dict, t)

        low_bat_cue_dict = {0: 'shallow', 1: 'border', 2: 'refugia'}
        try:
            self.is_in_eddy(t)
        except Exception:
            # If simulation lacks helpers during lightweight probes, skip eddy detection
            pass
        tolerance = 50000
        tol2 = float(tolerance) * float(tolerance)
        vec_sum_migratory = np.zeros_like(rheotaxis)
        vec_sum_tired = np.zeros_like(rheotaxis)

        cue_magnitudes = {} if want_record else None
        raw_vecs = {} if want_record else None
        if debug_behavior and not hasattr(self.simulation, 'last_cue_vecs'):
            self._safe_set_sim_attr('last_cue_vecs', {})

        # helper: coerce cue arrays to shape (num_agents, 2)
        n_agents = int(getattr(self.simulation, 'num_agents', 0) or 0)
        cap = float(getattr(self.simulation, 'max_cue_magnitude', 5000.0))
        cap2 = cap * cap
        for i in order_dict.keys():
            cue = order_dict[i]
            vec0 = cue_dict[cue]
            # coerce to (n_agents, 2) to avoid accidental broadcasting
            if isinstance(vec0, np.ndarray) and vec0.shape == (n_agents, 2):
                vec = vec0
            else:
                vec = self._coerce_agent_vec(vec0)
            vec = np.asarray(vec)

            # clip per-agent cue magnitudes to avoid single cue domination
            try:
                if cap > 0.0:
                    vx = vec[:, 0]
                    vy = vec[:, 1]
                    norms2 = vx * vx + vy * vy
                    over = norms2 > cap2
                    if np.any(over):
                        scale = cap / np.sqrt(norms2[over])
                        vec[over, 0] = vx[over] * scale
                        vec[over, 1] = vy[over] * scale
                        if debug_behavior:
                            try:
                                n_clip = int(np.sum(over))
                                if n_clip > 0:
                                    logging.getLogger(__name__).debug('DBG arbitrate: clipped %d agents for cue=%s (cap=%s)', n_clip, cue, cap)
                            except Exception:
                                pass
            except Exception:
                pass

            # Save coerced/clipped vector for later use.
            cue_dict[cue] = vec
            if want_record:
                raw_vecs[cue] = vec
                try:
                    cue_magnitudes[cue] = np.sqrt(vec[:, 0] * vec[:, 0] + vec[:, 1] * vec[:, 1])
                except Exception:
                    cue_magnitudes[cue] = np.zeros(n_agents, dtype=float)

            # Accumulate migratory sum for agents under tolerance (avoid per-cue np.where allocations).
            active = (vec_sum_migratory[:, 0] * vec_sum_migratory[:, 0] + vec_sum_migratory[:, 1] * vec_sum_migratory[:, 1]) < tol2
            if np.any(active):
                vec_sum_migratory[active] += vec[active]
        # debug prints (guarded) to reveal raw_vecs and cue_magnitudes
        if debug_behavior and want_record:
            logging.getLogger(__name__).debug('DBG RAWVECS POST BUILD keys=%s', list(raw_vecs.keys()))
            logging.getLogger(__name__).debug('DBG CUE_MAGS POST BUILD keys=%s', list(cue_magnitudes.keys()))

        # debug: show raw_vecs and cue_magnitudes available at this point
        if debug_behavior and want_record:
            try:
                kv = {k: (np.asarray(v).shape if hasattr(v, 'shape') else None) for k, v in raw_vecs.items()}
            except Exception:
                kv = {k: None for k in raw_vecs.keys()}
            try:
                km = {k: (np.asarray(v).shape if hasattr(v, 'shape') else None) for k, v in cue_magnitudes.items()}
            except Exception:
                km = {k: None for k in cue_magnitudes.keys()}
            try:
                logging.getLogger(__name__).debug('DBG raw_vecs keys/shapes=%s cue_magnitudes shapes=%s', kv, km)
            except Exception:
                pass

        if want_record:
            # persist rawvecs/magnitudes for downstream diagnostics tools (best-effort)
            self._safe_set_sim_attr('last_cue_vecs', {k: np.asarray(v) for k, v in raw_vecs.items()})
            # Optional NPZ dump of raw per-cue vectors and magnitudes for deterministic debugging.
            self._maybe_dump_rawvecs(t, raw_vecs, cue_magnitudes)

        for cue in ('shallow', 'border', 'refugia'):
            vec = cue_dict[cue]
            if isinstance(vec, np.ndarray) and vec.shape == (n_agents, 2):
                v2 = vec
            else:
                v2 = self._coerce_agent_vec(vec)
            active_t = (vec_sum_tired[:, 0] * vec_sum_tired[:, 0] + vec_sum_tired[:, 1] * vec_sum_tired[:, 1]) < tol2
            if np.any(active_t):
                vec_sum_tired[active_t] += np.asarray(v2)[active_t]

        head_vec = np.zeros_like(rheotaxis)
        swim_behav = np.asarray(self.simulation.swim_behav).reshape((-1,))
        mig = swim_behav == 1
        tired = (swim_behav == 2) | (swim_behav == 3)
        if np.any(mig):
            head_vec[mig] = vec_sum_migratory[mig]
        if np.any(tired):
            head_vec[tired] = vec_sum_tired[tired]

        # in-eddy override uses border+shallow (already coerced/clipped in cue_dict)
        try:
            in_eddy = np.asarray(self.simulation.in_eddy).reshape((-1,)) == 1
        except Exception:
            in_eddy = None
        if in_eddy is not None and np.any(in_eddy):
            head_vec[in_eddy] = cue_dict['border'][in_eddy] + cue_dict['shallow'][in_eddy]

        if debug_behavior:
            logging.getLogger(__name__).debug(
                'DBG arbitrate: head_vec.shape=%s debug_behavior=%s',
                getattr(head_vec, 'shape', None),
                debug_behavior,
            )

        if len(head_vec.shape) == 2:
            # Fast path: if we aren't recording diagnostics/state, return the new headings now.
            if not want_record:
                try:
                    return np.arctan2(head_vec[:, 1], head_vec[:, 0])
                except Exception:
                    return np.asarray(self.simulation.heading)
            # debug snapshot of cue magnitudes when debug_behavior is enabled
            if debug_behavior:
                try:
                    import json, time
                    # show mean, max, and nonzero counts per cue
                    cue_summary = {}
                    for k, v in cue_magnitudes.items():
                        arr = np.asarray(v, dtype=float)
                        nonzero = int(np.sum(np.isfinite(arr) & (np.abs(arr) > 0)))
                        mean = float(np.nanmean(arr)) if arr.size > 0 else float('nan')
                        mx = float(np.nanmax(arr)) if arr.size > 0 else float('nan')
                        cue_summary[k] = {'mean': mean, 'max': mx, 'nonzero_count': nonzero}

                    # include test_weights overview when present
                    tw = getattr(self.simulation, 'test_weights', None)

                    # enqueue a compact JSON snapshot for this step for later parsing
                    outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
                    try:
                        snap = {
                            'step': int(getattr(self.simulation, 'current_step', t)),
                            'time': int(time.time()),
                            'cue_summary': cue_summary,
                            'test_weights': tw if tw is not None else {},
                        }
                        self._enqueue_diag_json(outdir, f'behavior_cues_step_{int(getattr(self.simulation, "current_step", t))}', snap)
                    except Exception:
                        pass
                except Exception:
                    pass
            # store last head_vec and cue magnitudes on simulation for quick inspection
            try:
                self.simulation.last_head_vec = np.asarray(head_vec)
                self.simulation.last_cue_magnitudes = {k: np.asarray(v) for k, v in cue_magnitudes.items()}
                # persist raw per-cue vectors so external runners can include them in diagnostics
                try:
                    self.simulation.last_cue_vecs = {k: np.asarray(v) for k, v in raw_vecs.items()}
                except Exception:
                    # best-effort: skip if raw_vecs are not serializable
                    pass
            except Exception:
                pass

            # Robust final assignment: ensure attributes exist, correct shapes, and are serializable.
            n_agents = int(getattr(self.simulation, 'num_agents', 0)) or int(getattr(self.simulation, 'n_agents', 0))
            if n_agents <= 0:
                n_agents = int(getattr(self.simulation, 'num_agents', 0))

            # Build safe last_cue_vecs with guaranteed shape (n_agents,2)
            last_cue_vecs_final = {}
            for k, v in raw_vecs.items():
                try:
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.ndim == 1 and arr.size == 2:
                        arr = np.tile(arr.reshape(1, 2), (n_agents, 1))
                    if arr.ndim == 2 and arr.shape[0] == n_agents and arr.shape[1] == 2:
                        last_cue_vecs_final[k] = arr
                    else:
                        arr = arr.reshape((n_agents, 2)).astype(np.float32)
                        last_cue_vecs_final[k] = arr
                except Exception:
                    last_cue_vecs_final[k] = np.zeros((n_agents, 2), dtype=np.float32)

            # Ensure known cues are present even if empty.
            # Keep both `rheotaxis` (canonical) and `rheo` (legacy alias).
            known_cues = (
                'rheotaxis',
                'alignment',
                'cohesion',
                'low_speed',
                'wave_drag',
                'refugia',
                'border',
                'shallow',
                'avoid',
                'collision',
            )
            for known in known_cues:
                if known not in last_cue_vecs_final:
                    last_cue_vecs_final[known] = np.zeros((n_agents, 2), dtype=np.float32)
            if 'rheo' not in last_cue_vecs_final:
                last_cue_vecs_final['rheo'] = last_cue_vecs_final.get('rheotaxis', np.zeros((n_agents, 2), dtype=np.float32))

            try:
                self._safe_set_sim_attr('last_cue_vecs', last_cue_vecs_final)
            except Exception:
                pass

            # magnitudes
            last_cue_mags = {}
            for k, v in cue_magnitudes.items():
                try:
                    last_cue_mags[k] = np.asarray(v, dtype=np.float32)
                except Exception:
                    last_cue_mags[k] = np.zeros((n_agents,), dtype=np.float32)
            for known in known_cues:
                if known not in last_cue_mags:
                    last_cue_mags[known] = np.zeros((n_agents,), dtype=np.float32)
            if 'rheo' not in last_cue_mags:
                last_cue_mags['rheo'] = last_cue_mags.get('rheotaxis', np.zeros((n_agents,), dtype=np.float32))
            try:
                self._safe_set_sim_attr('last_cue_magnitudes', last_cue_mags)
            except Exception:
                pass

            # head vector
            try:
                hv = np.asarray(head_vec, dtype=np.float32)
                if hv.ndim == 1 and hv.size == 2:
                    hv = np.tile(hv.reshape(1, 2), (n_agents, 1))
                hv = hv.reshape((n_agents, 2)).astype(np.float32)
            except Exception:
                hv = np.zeros((n_agents, 2), dtype=np.float32)
            try:
                self._safe_set_sim_attr('last_head_vec', hv)
            except Exception:
                pass

            try:
                step_i = int(getattr(self.simulation, 'current_step', t))
                payload = {}
                try:
                    payload['head_vec'] = np.asarray(hv).astype(float)
                except Exception:
                    payload['head_vec'] = np.zeros((n_agents if n_agents else 0, 2), dtype=float)
                for k, v in last_cue_vecs_final.items():
                    try:
                        payload[f'{k}_vec'] = np.asarray(v).astype(float)
                    except Exception:
                        payload[f'{k}_vec'] = np.zeros((n_agents if n_agents else 0, 2), dtype=float)

                # attempt HDF5 diagnostics writer, fall back to NPZ if configured
                try:
                    self._safe_write_diagnostics(step_i, payload)
                except Exception:
                    pass
            except Exception:
                # non-fatal: skip diagnostics on any failure
                pass

            # optional behavior debugging: simplified dump
            if debug_behavior:
                try:
                    outdir = getattr(self.simulation, 'model_dir', None) or os.path.join('outputs', 'diagnostics')
                    os.makedirs(outdir, exist_ok=True)
                    step_i = int(getattr(self.simulation, 'current_step', t))
                    payload = {}
                    try:
                        payload['head_vec'] = np.asarray(head_vec).astype(float)
                    except Exception:
                        payload['head_vec'] = np.zeros((getattr(self.simulation, 'num_agents', 0) or 0, 2), dtype=float)
                    try:
                        for k, v in cue_magnitudes.items():
                            payload[f'{k}_mag'] = np.asarray(v).astype(float)
                    except Exception:
                        pass
                    try:
                        self._safe_write_diagnostics(step_i, payload, outdir=outdir)
                    except Exception:
                        # best-effort: enqueue small JSON if NPZ/HDF5 fails
                        try:
                            serial = {'head_vec': payload.get('head_vec', []).tolist(), 'cue_magnitudes': {k: v.tolist() for k, v in payload.items() if k.endswith('_mag')}}
                            self._enqueue_diag_json(outdir, f'behavior_headvecs_step_{step_i}', serial)
                        except Exception:
                            pass
                except Exception:
                    try:
                        logging.getLogger(__name__).debug('simplified behavior debug dump failed')
                    except Exception:
                        pass
            return np.arctan2(head_vec[:, 1], head_vec[:, 0])
        else:
            # If head_vec has unexpected shape, try to sanitize: replace NaNs and zero-length vectors
            try:
                hv = np.asarray(head_vec)
                if hv.ndim == 3:
                    hv = hv.reshape(hv.shape[0], -1)
                # compute norms and replace NaNs/zeros
                norms = np.linalg.norm(hv, axis=-1)
                # fallback unit vectors from previous headings
                prev_hat_x = np.cos(self.simulation.heading)
                prev_hat_y = np.sin(self.simulation.heading)
                prev_hat = np.column_stack((prev_hat_x, prev_hat_y))
                # where norm is zero or nan, replace with prev_hat or rheotaxis
                safe_hv = np.where(np.isnan(norms)[:, np.newaxis] | (norms[:, np.newaxis] == 0), prev_hat, hv)
                # Ensure we also persist a safe last_head_vec and cue vecs even in this fallback path
                try:
                    n_agents = int(getattr(self.simulation, 'num_agents', 0)) or int(getattr(self.simulation, 'n_agents', 0))
                except Exception:
                    n_agents = getattr(self.simulation, 'num_agents', None) or getattr(self.simulation, 'n_agents', None) or 0
                try:
                    if n_agents and getattr(self.simulation, 'last_cue_vecs', None) is None:
                        # build minimal last_cue_vecs from raw_vecs if available
                        try:
                            last_cue_vecs_final = {k: np.zeros((n_agents, 2), dtype=np.float32) for k in ('cohesion', 'alignment', 'rheo', 'refugia', 'border', 'shallow', 'collision', 'avoid')}
                            if 'raw_vecs' in locals():
                                for k, v in raw_vecs.items():
                                    try:
                                        arr = np.asarray(v, dtype=np.float32)
                                        if arr.ndim == 1 and arr.size == 2:
                                            arr = np.tile(arr.reshape(1, 2), (n_agents, 1))
                                        if arr.ndim == 2 and arr.shape[0] == n_agents and arr.shape[1] == 2:
                                            last_cue_vecs_final[k] = arr
                                    except Exception:
                                        pass
                            self.simulation.last_cue_vecs = last_cue_vecs_final
                        except Exception:
                            try:
                                setattr(self.simulation, 'last_cue_vecs', {})
                            except Exception:
                                pass
                    # set last_head_vec to safe_hv coerced
                    try:
                        hv_safe = np.asarray(safe_hv, dtype=np.float32)
                        if hv_safe.ndim == 1 and hv_safe.size == 2:
                            hv_safe = np.tile(hv_safe.reshape(1, 2), (n_agents if n_agents else 1, 1))
                        self.simulation.last_head_vec = hv_safe
                    except Exception:
                        try:
                            setattr(self.simulation, 'last_head_vec', np.zeros((n_agents if n_agents else 1, 2), dtype=np.float32))
                        except Exception:
                            pass
                except Exception:
                    pass
                return np.arctan2(safe_hv[:, 1], safe_hv[:, 0])
            except Exception:
                # ultimate fallback: return previous heading
                return np.asarray(self.simulation.heading)
        # end of arbitrate: handled 2D and attempted safe fallback above
