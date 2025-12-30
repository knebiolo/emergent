"""Acceptance checks for Salmon ABM cue directionality.

Goal: for each isolated cue, verify that force vectors point toward (attractive)
or away from (repulsive) the intended target feature.

This script is intentionally lightweight: it prints a concise per-cue summary and
optionally writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io, io
from emergent.salmon_abm.utils import geo_to_pixel, pixel_to_geo


@dataclass(frozen=True)
class CueCase:
    name: str
    start_polygon: str
    polarity: str  # "attractive" or "repulsive"


def _data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "salmon_abm"))


def discover_env_files(base_dir: str) -> list[str]:
    keys = ["depth.tif", "vel_x.tif", "vel_y.tif", "vel_mag.tif", "vel_dir.tif"]
    out: list[str] = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def import_env_to_h5(sim: simulation, env_files: list[str]) -> None:
    h5 = hdf5_io.get_hdf5_obj(sim)
    for ef in env_files:
        try:
            io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
        except Exception:
            continue

    # write x/y coordinate grids (behavior expects these for some cues)
    try:
        depth_ds = hdf5_io.read_dataset(h5, "environment/depth", default=None)
        if depth_ds is None:
            return
        depth_arr = np.asarray(depth_ds)
        if depth_arr.ndim != 2:
            return
        nrows, ncols = depth_arr.shape
        a, b, c, d, e, f = getattr(sim, "depth_rast_transform", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        cols = np.arange(ncols, dtype=float)
        rows = np.arange(nrows, dtype=float)
        col_indices, row_indices = np.meshgrid(cols, rows)
        x_coords = a * col_indices + b * row_indices + c
        y_coords = d * col_indices + e * row_indices + f
        hdf5_io.write_dataset(h5, "environment/x_coords", x_coords)
        hdf5_io.write_dataset(h5, "environment/y_coords", y_coords)
    except Exception:
        return


def ensure_distance_to(sim: simulation) -> bool:
    h5 = hdf5_io.get_hdf5_obj(sim)
    existing = hdf5_io.read_dataset(h5, "environment/distance_to", default=None)
    if existing is not None:
        try:
            arr = np.asarray(existing)
            if arr.ndim == 2 and arr.size > 1:
                return True
        except Exception:
            return True

    depth = hdf5_io.read_dataset(h5, "environment/depth", default=None)
    if depth is None:
        return False
    depth_arr = np.asarray(depth, dtype=float)
    if depth_arr.ndim != 2 or depth_arr.size <= 1:
        return False
    wetted = np.isfinite(depth_arr) & (depth_arr != -9999.0)
    try:
        tr = getattr(sim, "depth_rast_transform", None)
        pw = float(tr[0]) if tr is not None else 1.0
    except Exception:
        pw = 1.0
    dist_to_bound = distance_transform_edt(wetted) * abs(pw)
    hdf5_io.write_dataset(h5, "environment/distance_to", dist_to_bound.astype("float32"))
    return True


def ensure_refugia_layer(sim: simulation, velmag_threshold: float) -> bool:
    h5 = hdf5_io.get_hdf5_obj(sim)
    existing = hdf5_io.read_dataset(h5, "environment/refugia", default=None)
    if existing is not None:
        try:
            arr = np.asarray(existing)
            if arr.ndim == 2 and arr.size > 1:
                return True
        except Exception:
            return True

    vel_mag = hdf5_io.read_dataset(h5, "environment/vel_mag", default=None)
    if vel_mag is None:
        return False
    vel_mag_arr = np.asarray(vel_mag, dtype=float)
    if vel_mag_arr.ndim != 2 or vel_mag_arr.size <= 1:
        return False
    finite = np.isfinite(vel_mag_arr) & (vel_mag_arr != -9999.0)
    mask = finite & (vel_mag_arr <= float(velmag_threshold))
    # fallback: if explicit threshold produces no refugia, use a low quantile so cue is testable
    if not np.any(mask) and np.any(finite):
        try:
            qthr = float(np.nanpercentile(vel_mag_arr[finite], 10))
            mask = finite & (vel_mag_arr <= qthr)
        except Exception:
            pass
    refugia = np.zeros_like(vel_mag_arr, dtype=np.uint8)
    refugia[mask] = 1
    hdf5_io.write_dataset(h5, "environment/refugia", refugia)
    return True


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = an * bn
    out = np.full(a.shape[0], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / denom[ok]
    return out


def _front_mask(dx: np.ndarray, dy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    hx = np.cos(heading)
    hy = np.sin(heading)
    return (dx * hx + dy * hy) > 0


def _expected_rheotaxis(sim: simulation, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    tx = getattr(sim, "vel_x_rast_transform", None) or getattr(sim, "depth_rast_transform", None)
    ty = getattr(sim, "vel_y_rast_transform", None) or getattr(sim, "depth_rast_transform", None)
    # sample environment at current positions
    vx = np.asarray(sim.sample_environment(tx, "vel_x"), dtype=float)
    vy = np.asarray(sim.sample_environment(ty, "vel_y"), dtype=float)
    v = np.column_stack((-vx, -vy))  # upstream
    mags = np.linalg.norm(v, axis=1)
    mags_safe = np.where(mags == 0, 1.0, mags)
    out = v / mags_safe[:, None]
    out[~np.isfinite(out)] = 0.0
    return out


def _expected_window_target(sim: simulation, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray, ds_name: str, score_fn) -> np.ndarray:
    h5 = hdf5_io.get_hdf5_obj(sim)
    ds = hdf5_io.read_dataset(h5, f"environment/{ds_name}", default=None)
    if ds is None:
        return np.zeros((sim.num_agents, 2), dtype=float)
    arr = np.asarray(ds, dtype=float)
    if arr.ndim != 2:
        return np.zeros((sim.num_agents, 2), dtype=float)

    transform = getattr(sim, "depth_rast_transform", None)
    rows, cols = geo_to_pixel(x0, y0, transform)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    buff = 2

    out = np.zeros((sim.num_agents, 2), dtype=float)
    for i in range(sim.num_agents):
        r0 = max(0, int(rows[i]) - buff)
        r1 = min(arr.shape[0], int(rows[i]) + buff + 1)
        c0 = max(0, int(cols[i]) - buff)
        c1 = min(arr.shape[1], int(cols[i]) + buff + 1)
        window = arr[r0:r1, c0:c1]
        if window.size == 0:
            continue
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        xs, ys = pixel_to_geo(transform, rr, cc)
        dx = np.asarray(xs, dtype=float) - float(x0[i])
        dy = np.asarray(ys, dtype=float) - float(y0[i])
        front = _front_mask(dx, dy, float(heading0[i]))
        # if nothing is in front, fall back to the entire window
        use = front if np.any(front) else np.ones_like(front, dtype=bool)
        scores = score_fn(window)
        scores = np.where(use, scores, np.inf)
        if not np.isfinite(scores).any():
            continue
        j = int(np.nanargmin(scores))
        wr, wc = np.unravel_index(j, scores.shape)
        tgt_x = float(xs[wr, wc])
        tgt_y = float(ys[wr, wc])
        ddx = tgt_x - float(x0[i])
        ddy = tgt_y - float(y0[i])
        dist = (ddx * ddx + ddy * ddy) ** 0.5
        if dist <= 0 or not np.isfinite(dist):
            continue
        out[i, 0] = ddx / dist
        out[i, 1] = ddy / dist
    return out


def _expected_low_speed(sim: simulation, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray) -> np.ndarray:
    def score(w):
        w = np.asarray(w, dtype=float)
        return np.where(np.isfinite(w) & (w > -9990.0), w, np.inf)

    return _expected_window_target(sim, x0, y0, heading0, "vel_mag", score)


def _expected_wave_drag(sim: simulation, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray) -> np.ndarray:
    opt = np.asarray(getattr(sim, "opt_wat_depth", np.zeros(sim.num_agents)), dtype=float)

    def score(w):
        # w: (H,W); opt per agent is handled outside by closure not possible; compute later
        return w

    # custom per-agent scoring to avoid copying large windows
    h5 = hdf5_io.get_hdf5_obj(sim)
    ds = hdf5_io.read_dataset(h5, "environment/depth", default=None)
    if ds is None:
        return np.zeros((sim.num_agents, 2), dtype=float)
    arr = np.asarray(ds, dtype=float)
    if arr.ndim != 2:
        return np.zeros((sim.num_agents, 2), dtype=float)

    transform = getattr(sim, "depth_rast_transform", None)
    rows, cols = geo_to_pixel(x0, y0, transform)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    buff = 2

    out = np.zeros((sim.num_agents, 2), dtype=float)
    for i in range(sim.num_agents):
        r0 = max(0, int(rows[i]) - buff)
        r1 = min(arr.shape[0], int(rows[i]) + buff + 1)
        c0 = max(0, int(cols[i]) - buff)
        c1 = min(arr.shape[1], int(cols[i]) + buff + 1)
        window = arr[r0:r1, c0:c1]
        if window.size == 0:
            continue
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        xs, ys = pixel_to_geo(transform, rr, cc)
        dx = np.asarray(xs, dtype=float) - float(x0[i])
        dy = np.asarray(ys, dtype=float) - float(y0[i])
        front = _front_mask(dx, dy, float(heading0[i]))
        use = front if np.any(front) else np.ones_like(front, dtype=bool)
        window = np.asarray(window, dtype=float)
        window = np.where(np.isfinite(window) & (window > -9990.0), window, np.inf)
        scores = np.abs(window - float(opt[i]))
        scores = np.where(use, scores, np.inf)
        if not np.isfinite(scores).any():
            continue
        j = int(np.nanargmin(scores))
        wr, wc = np.unravel_index(j, scores.shape)
        tgt_x = float(xs[wr, wc])
        tgt_y = float(ys[wr, wc])
        ddx = tgt_x - float(x0[i])
        ddy = tgt_y - float(y0[i])
        dist = (ddx * ddx + ddy * ddy) ** 0.5
        if dist <= 0 or not np.isfinite(dist):
            continue
        out[i, 0] = ddx / dist
        out[i, 1] = ddy / dist
    return out


def _expected_border(sim: simulation, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray) -> np.ndarray:
    h5 = hdf5_io.get_hdf5_obj(sim)
    dist = hdf5_io.read_dataset(h5, "environment/distance_to", default=None)
    if dist is None:
        return np.zeros((sim.num_agents, 2), dtype=float)
    arr = np.asarray(dist, dtype=float)
    if arr.ndim != 2:
        return np.zeros((sim.num_agents, 2), dtype=float)

    transform = getattr(sim, "depth_rast_transform", None)
    rows, cols = geo_to_pixel(x0, y0, transform)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    buff = 2

    # gate: only evaluate where "too_close" would be true
    cur_d = np.asarray(sim.sample_environment(transform, "distance_to"), dtype=float)
    length_m = np.asarray(sim.length, dtype=float) / 1000.0
    try:
        tr = getattr(sim, "depth_rast_transform", None)
        pw = abs(float(tr[0])) if tr is not None else 0.0
    except Exception:
        pw = 0.0
    influence_dist = np.maximum(10.0, np.maximum(10.0 * length_m, pw))
    too_close = np.isfinite(cur_d) & (cur_d <= influence_dist)

    out = np.zeros((sim.num_agents, 2), dtype=float)
    for i in range(sim.num_agents):
        if not bool(too_close[i]):
            continue
        r0 = max(0, int(rows[i]) - buff)
        r1 = min(arr.shape[0], int(rows[i]) + buff + 1)
        c0 = max(0, int(cols[i]) - buff)
        c1 = min(arr.shape[1], int(cols[i]) + buff + 1)
        window = arr[r0:r1, c0:c1]
        if window.size == 0:
            continue
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        xs, ys = pixel_to_geo(transform, rr, cc)
        dx = np.asarray(xs, dtype=float) - float(x0[i])
        dy = np.asarray(ys, dtype=float) - float(y0[i])
        front = _front_mask(dx, dy, float(heading0[i]))
        use = front if np.any(front) else np.ones_like(front, dtype=bool)
        scores = np.where(use, window, -np.inf)
        if not np.isfinite(scores).any():
            continue
        j = int(np.nanargmax(scores))
        wr, wc = np.unravel_index(j, scores.shape)
        tgt_x = float(xs[wr, wc])
        tgt_y = float(ys[wr, wc])
        ddx = tgt_x - float(x0[i])
        ddy = tgt_y - float(y0[i])
        distw = (ddx * ddx + ddy * ddy) ** 0.5
        if distw <= 0 or not np.isfinite(distw):
            continue
        out[i, 0] = ddx / distw
        out[i, 1] = ddy / distw
    return out


def _expected_shallow(sim: simulation, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray) -> np.ndarray:
    h5 = hdf5_io.get_hdf5_obj(sim)
    depth = hdf5_io.read_dataset(h5, "environment/depth", default=None)
    if depth is None:
        return np.zeros((sim.num_agents, 2), dtype=float)
    arr = np.asarray(depth, dtype=float)
    if arr.ndim != 2:
        return np.zeros((sim.num_agents, 2), dtype=float)

    transform = getattr(sim, "depth_rast_transform", None)
    rows, cols = geo_to_pixel(x0, y0, transform)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    buff = 2
    min_depth = np.asarray(getattr(sim, "too_shallow", np.zeros(sim.num_agents)), dtype=float)

    out = np.zeros((sim.num_agents, 2), dtype=float)
    for i in range(sim.num_agents):
        r0 = max(0, int(rows[i]) - buff)
        r1 = min(arr.shape[0], int(rows[i]) + buff + 1)
        c0 = max(0, int(cols[i]) - buff)
        c1 = min(arr.shape[1], int(cols[i]) + buff + 1)
        window = arr[r0:r1, c0:c1]
        if window.size == 0:
            continue
        rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        xs, ys = pixel_to_geo(transform, rr, cc)
        dx = float(x0[i]) - np.asarray(xs, dtype=float)
        dy = float(y0[i]) - np.asarray(ys, dtype=float)
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.where(dist == 0, 1e-6, dist)
        front = _front_mask(-dx, -dy, float(heading0[i]))
        shallow = window < float(min_depth[i])
        use = shallow & front
        if not np.any(use):
            continue
        # repulsive: sum delta/(dist^2) over shallow cells in front
        fx = np.sum((dx[use] / (dist_safe[use] ** 2)))
        fy = np.sum((dy[use] / (dist_safe[use] ** 2)))
        mag = (fx * fx + fy * fy) ** 0.5
        if mag <= 0 or not np.isfinite(mag):
            continue
        out[i, 0] = fx / mag
        out[i, 1] = fy / mag
    return out


def _expected_cohesion(sim: simulation, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    out = np.zeros((sim.num_agents, 2), dtype=float)
    awb = getattr(sim, "agents_within_buffers", None)
    if awb is None:
        return out
    for i in range(sim.num_agents):
        nbrs = awb[i]
        if nbrs is None or len(nbrs) == 0:
            continue
        cx = float(np.mean(x0[nbrs]))
        cy = float(np.mean(y0[nbrs]))
        dx = cx - float(x0[i])
        dy = cy - float(y0[i])
        mag = (dx * dx + dy * dy) ** 0.5
        if mag <= 0 or not np.isfinite(mag):
            continue
        out[i, 0] = dx / mag
        out[i, 1] = dy / mag
    return out


def _expected_collision(sim: simulation, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    out = np.zeros((sim.num_agents, 2), dtype=float)
    closest = np.asarray(getattr(sim, "closest_agent", np.full(sim.num_agents, np.nan)), dtype=float)
    for i in range(sim.num_agents):
        j = closest[i]
        if not np.isfinite(j):
            continue
        j = int(j)
        dx = float(x0[i]) - float(x0[j])
        dy = float(y0[i]) - float(y0[j])
        mag = (dx * dx + dy * dy) ** 0.5
        if mag <= 0 or not np.isfinite(mag):
            continue
        out[i, 0] = dx / mag
        out[i, 1] = dy / mag
    return out


def _expected_alignment(sim: simulation, heading0: np.ndarray) -> np.ndarray:
    out = np.zeros((sim.num_agents, 2), dtype=float)
    awb = getattr(sim, "agents_within_buffers", None)
    if awb is None:
        return out
    for i in range(sim.num_agents):
        nbrs = awb[i]
        if nbrs is None or len(nbrs) == 0:
            continue
        hs = np.asarray(heading0[nbrs], dtype=float)
        mx = float(np.mean(np.cos(hs)))
        my = float(np.mean(np.sin(hs)))
        mag = (mx * mx + my * my) ** 0.5
        if mag <= 0 or not np.isfinite(mag):
            continue
        out[i, 0] = mx / mag
        out[i, 1] = my / mag
    return out


def _expected_refugia(sim: simulation, x0: np.ndarray, y0: np.ndarray) -> np.ndarray:
    h5 = hdf5_io.get_hdf5_obj(sim)
    ref = hdf5_io.read_dataset(h5, "environment/refugia", default=None)
    if ref is None:
        return np.zeros((sim.num_agents, 2), dtype=float)
    arr = np.asarray(ref)
    if arr.ndim != 2:
        return np.zeros((sim.num_agents, 2), dtype=float)

    transform = getattr(sim, "depth_rast_transform", None)
    refuge_mask = (arr == 1)
    if not np.any(refuge_mask):
        return np.zeros((sim.num_agents, 2), dtype=float)

    try:
        _, inds = distance_transform_edt(~refuge_mask, return_indices=True)
        rows, cols = geo_to_pixel(x0, y0, transform)
        rows = np.clip(np.asarray(rows, dtype=int), 0, arr.shape[0] - 1)
        cols = np.clip(np.asarray(cols, dtype=int), 0, arr.shape[1] - 1)
        ref_r = np.asarray(inds[0])[rows, cols]
        ref_c = np.asarray(inds[1])[rows, cols]
        ref_x, ref_y = pixel_to_geo(transform, ref_r, ref_c)
        dx = np.asarray(ref_x, dtype=float) - np.asarray(x0, dtype=float)
        dy = np.asarray(ref_y, dtype=float) - np.asarray(y0, dtype=float)
        mag = np.sqrt(dx * dx + dy * dy)
        mag_safe = np.where(mag == 0, 1.0, mag)
        out = np.zeros((sim.num_agents, 2), dtype=float)
        out[:, 0] = dx / mag_safe
        out[:, 1] = dy / mag_safe
        out[~np.isfinite(out)] = 0.0
        out[mag == 0] = 0.0
        return out
    except Exception:
        return np.zeros((sim.num_agents, 2), dtype=float)


def expected_direction(sim: simulation, cue: str, x0: np.ndarray, y0: np.ndarray, heading0: np.ndarray) -> np.ndarray:
    cue = str(cue)
    if cue == "rheotaxis":
        return _expected_rheotaxis(sim, x0, y0)
    if cue == "low_speed":
        return _expected_low_speed(sim, x0, y0, heading0)
    if cue == "wave_drag":
        return _expected_wave_drag(sim, x0, y0, heading0)
    if cue == "border":
        return _expected_border(sim, x0, y0, heading0)
    if cue == "shallow":
        return _expected_shallow(sim, x0, y0, heading0)
    if cue == "cohesion":
        return _expected_cohesion(sim, x0, y0)
    if cue == "collision":
        return _expected_collision(sim, x0, y0)
    if cue == "alignment":
        return _expected_alignment(sim, heading0)
    if cue == "refugia":
        return _expected_refugia(sim, x0, y0)
    if cue == "avoid":
        try:
            exp = getattr(sim, "_avoid_expected_unit", None)
            if exp is not None:
                exp = np.asarray(exp, dtype=float)
                if exp.shape == (sim.num_agents, 2):
                    return exp
        except Exception:
            pass
    return np.zeros((sim.num_agents, 2), dtype=float)


def run_case(case: CueCase, *, outdir: str, nagents: int, dt: float, weight: float, seed: int | None, derive_refugia: bool, refugia_velmag_threshold: float) -> dict:
    base = _data_dir()
    env_files = discover_env_files(base)

    sim = simulation(
        model_dir=outdir,
        model_name=f"accept_{case.name}",
        crs=None,
        basin="nuyakuk",
        water_temp=10.0,
        start_polygon=case.start_polygon if os.path.exists(case.start_polygon) else None,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, "longitudinal.shp") if os.path.exists(os.path.join(base, "longitudinal.shp")) else None,
        num_timesteps=2,
        num_agents=nagents,
        db_path=os.path.join(outdir, f"accept_{case.name}.h5"),
    )

    # deterministic placement and any random choices (best-effort)
    if seed is not None:
        try:
            sim.rng = np.random.default_rng(int(seed))
        except Exception:
            pass
    sim.neighbor_buffer_radius = float(getattr(sim, "neighbor_buffer_radius", 10.0)) * 5.0
    sim.max_cue_magnitude = 1e12

    import_env_to_h5(sim, env_files)
    ensure_distance_to(sim)
    if derive_refugia:
        ensure_refugia_layer(sim, refugia_velmag_threshold)

    try:
        sim.initialize_headings_from_db()
    except Exception:
        pass

    # isolate cue under test
    sim.test_weights = {case.name: float(weight)}

    x0 = np.asarray(sim.X, dtype=float).copy()
    y0 = np.asarray(sim.Y, dtype=float).copy()
    heading0 = np.asarray(sim.heading, dtype=float).copy()

    # Special-case: avoid cue requires preseeded memory timestamps and a timestep > 10
    t0 = 0.0
    if case.name == "avoid":
        try:
            sim.initialize_mental_map()
            h5 = hdf5_io.get_hdf5_obj(sim)
            try:
                mh, mw = h5["memory/0"].shape
            except Exception:
                mh, mw = (1, 1)
            rows, cols = geo_to_pixel(x0, y0, getattr(sim, "mental_map_transform"))
            rows = np.asarray(rows, dtype=int)
            cols = np.asarray(cols, dtype=int)
            # seed one visited cell to the east of each agent
            seed_r = np.clip(rows, 0, int(mh) - 1)
            seed_c = np.clip(cols + 1, 0, int(mw) - 1)
            # compute expected direction (away from the seeded cell)
            seed_x, seed_y = pixel_to_geo(getattr(sim, "mental_map_transform"), seed_r, seed_c)
            dx = x0 - np.asarray(seed_x, dtype=float)
            dy = y0 - np.asarray(seed_y, dtype=float)
            mag = np.sqrt(dx * dx + dy * dy)
            mag_safe = np.where(mag == 0, 1.0, mag)
            sim._avoid_expected_unit = np.column_stack((dx / mag_safe, dy / mag_safe))
            # write timestamps into memory maps
            for i in range(sim.num_agents):
                try:
                    ds = h5[f"memory/{i}"]
                    rr = int(np.clip(seed_r[i], 0, ds.shape[0] - 1))
                    cc = int(np.clip(seed_c[i], 0, ds.shape[1] - 1))
                    ds[rr, cc] = 0.0
                except Exception:
                    pass
        except Exception:
            pass
        t0 = 20.0

    sim.timestep(float(t0), float(dt))

    cue_vecs = getattr(sim, "last_cue_vecs", {}) or {}
    cue_vec = np.asarray(cue_vecs.get(case.name, np.zeros((nagents, 2))), dtype=float)
    exp = expected_direction(sim, case.name, x0, y0, heading0)

    cos = _cosine(cue_vec, exp)
    valid = np.isfinite(cos)

    summary = {
        "cue": case.name,
        "polarity": case.polarity,
        "start_polygon": os.path.basename(case.start_polygon),
        "nagents": int(nagents),
        "nonzero_frac": float(np.mean(np.linalg.norm(cue_vec, axis=1) > 0)),
        "valid_frac": float(np.mean(valid)),
    }
    if np.any(valid):
        summary.update(
            {
                "mean_cos": float(np.nanmean(cos)),
                "pos_frac": float(np.mean(cos[valid] > 0)),
                "p10_cos": float(np.nanpercentile(cos[valid], 10)),
            }
        )
    else:
        summary.update({"mean_cos": float("nan"), "pos_frac": float("nan"), "p10_cos": float("nan")})

    # simple pass/fail: sign must be consistent for the majority of valid agents
    summary["pass"] = bool(summary["pos_frac"] >= 0.9) if np.any(valid) else False

    try:
        sim.close()
    except Exception:
        pass
    return summary


def run_fatigue_check(*, outdir: str, nagents: int, nsteps: int, dt: float, seed: int | None) -> dict:
    base = _data_dir()
    env_files = discover_env_files(base)
    start_polygon = os.path.join(base, "at_falls.shp")
    sim = simulation(
        model_dir=outdir,
        model_name="accept_fatigue",
        crs=None,
        basin="nuyakuk",
        water_temp=10.0,
        start_polygon=start_polygon if os.path.exists(start_polygon) else None,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, "longitudinal.shp") if os.path.exists(os.path.join(base, "longitudinal.shp")) else None,
        num_timesteps=max(2, int(nsteps) + 1),
        num_agents=nagents,
        db_path=os.path.join(outdir, "accept_fatigue.h5"),
    )
    if seed is not None:
        try:
            sim.rng = np.random.default_rng(int(seed))
        except Exception:
            pass

    import_env_to_h5(sim, env_files)
    ensure_distance_to(sim)
    try:
        sim.initialize_headings_from_db()
    except Exception:
        pass

    # activate low_speed only (per your acceptance focus) and allow rheotaxis to remain 0
    sim.test_weights = {"low_speed": 3000.0}

    # record battery evolution
    b0 = np.asarray(getattr(sim, "battery", np.ones(nagents)), dtype=float).copy()
    for t in range(int(nsteps)):
        sim.timestep(t, float(dt))
    b1 = np.asarray(getattr(sim, "battery", np.ones(nagents)), dtype=float).copy()
    sb = np.asarray(getattr(sim, "swim_behav", np.ones(nagents)), dtype=int).copy()

    out = {
        "fatigue_steps": int(nsteps),
        "battery_mean_start": float(np.nanmean(b0)),
        "battery_mean_end": float(np.nanmean(b1)),
        "battery_min_end": float(np.nanmin(b1)),
        "swim_behav_counts_end": {
            "1": int(np.sum(sb == 1)),
            "2": int(np.sum(sb == 2)),
            "3": int(np.sum(sb == 3)),
        },
    }
    try:
        sim.close()
    except Exception:
        pass
    return out


def main() -> None:
    base = _data_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("outputs", "acceptance"))
    parser.add_argument("--nagents", type=int, default=300)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--weight", type=float, default=50000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", dest="json_path", default=None, help="Optional JSON output path")
    parser.add_argument("--derive-refugia", action="store_true", help="Derive environment/refugia from vel_mag")
    parser.add_argument("--refugia-velmag-threshold", type=float, default=0.3)
    parser.add_argument("--fatigue-steps", type=int, default=200)
    args = parser.parse_args()

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    start_river_right = os.path.join(base, "start_loc_river_right.shp")
    start_near_shore = os.path.join(base, "near_shore.shp")

    cases = [
        # schooling: river right
        CueCase("collision", start_river_right, "repulsive"),
        CueCase("alignment", start_river_right, "attractive"),
        CueCase("cohesion", start_river_right, "attractive"),
        # near shore: near_shore
        CueCase("refugia", start_near_shore, "attractive"),
        CueCase("border", start_near_shore, "repulsive"),
        CueCase("shallow", start_near_shore, "repulsive"),
        CueCase("avoid", start_near_shore, "repulsive"),
        CueCase("wave_drag", start_near_shore, "attractive"),
        # global attractive cue (not tied to a start polygon choice in the request)
        CueCase("rheotaxis", start_river_right, "attractive"),
        CueCase("low_speed", os.path.join(base, "at_falls.shp"), "attractive"),
    ]

    results = []
    for case in cases:
        results.append(
            run_case(
                case,
                outdir=outdir,
                nagents=int(args.nagents),
                dt=float(args.dt),
                weight=float(args.weight),
                seed=int(args.seed) if args.seed is not None else None,
                derive_refugia=bool(args.derive_refugia),
                refugia_velmag_threshold=float(args.refugia_velmag_threshold),
            )
        )

    fatigue = run_fatigue_check(outdir=outdir, nagents=int(args.nagents), nsteps=int(args.fatigue_steps), dt=float(args.dt), seed=int(args.seed) if args.seed is not None else None)

    # concise console output
    for r in results:
        print(
            f"{r['cue']}: pass={r['pass']} nonzero={r['nonzero_frac']:.3f} valid={r['valid_frac']:.3f} "
            f"mean_cos={r['mean_cos']:.3f} pos_frac={r['pos_frac']:.3f} ({r['start_polygon']})"
        )
    print("fatigue:", json.dumps(fatigue, indent=2))

    def _json_sanitize(v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return None
            return v
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return fv if np.isfinite(fv) else None
        if isinstance(v, dict):
            return {k: _json_sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_sanitize(x) for x in v]
        return v

    report = {"cue_results": results, "fatigue": fatigue}
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(_json_sanitize(report), fh, indent=2, allow_nan=False)
        print("wrote:", os.path.abspath(args.json_path))


if __name__ == "__main__":
    main()
