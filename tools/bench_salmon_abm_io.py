"""Benchmark Salmon ABM timestep wallclock under different output backends.

This focuses on end-to-end timing (not cProfile) so we can compare:
- sync HDF writes (full/minimal/none)
- async thread writer
- async process writer

Example:
  python tools/bench_salmon_abm_io.py --nagents 2000 --steps 200 --backend sync --mode none --tag base
  python tools/bench_salmon_abm_io.py --nagents 2000 --steps 200 --backend process --mode none --keys agent_data/X,agent_data/Y --tag proc_xy
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np

from emergent.salmon_abm import simulation as simmod
from emergent.salmon_abm import hdf5_io, io as salmon_io

logger = logging.getLogger(__name__)


def _data_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "salmon_abm"))


def _discover_env_files(base_dir: str) -> list[str]:
    keys = ["depth.tif", "vel_x.tif", "vel_y.tif", "vel_mag.tif", "vel_dir.tif"]
    out: list[str] = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def _parse_keys(s: str | None) -> tuple[str, ...] | None:
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",")]
    parts = [p for p in parts if p]
    return tuple(parts) if parts else None


def _prime_env_cache(sim) -> None:
    for k in (
        "environment/depth",
        "environment/vel_x",
        "environment/vel_y",
        "environment/vel_mag",
        "environment/vel_dir",
        "environment/distance_to",
        "environment/refugia",
        "environment/x_coords",
        "environment/y_coords",
    ):
        try:
            sim.get_cached_dataset(k, default=None)
        except Exception:
            logger.debug("Failed priming env cache for %s", k, exc_info=True)


def _run_one(
    *,
    nagents: int,
    steps: int,
    dt: float,
    warmup_steps: int,
    backend: str,
    mode: str,
    write_frequency: int,
    keys: tuple[str, ...] | None,
    tag: str,
    outdir: str,
) -> dict:
    base = _data_dir()
    env_files = _discover_env_files(base)
    start_poly = os.path.join(base, "near_shore.shp")
    if not os.path.exists(start_poly):
        start_poly = None

    os.makedirs(outdir, exist_ok=True)
    db_path = os.path.join(outdir, f"bench_{tag}_{backend}_{mode}_{nagents}_{steps}.h5")

    sim = simmod.simulation(
        model_dir=outdir,
        model_name=f"bench_{tag}",
        crs=None,
        basin="nuyakuk",
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, "longitudinal.shp")
        if os.path.exists(os.path.join(base, "longitudinal.shp"))
        else None,
        num_timesteps=max(1, int(steps) + int(warmup_steps) + 5),
        num_agents=int(nagents),
        db_path=db_path,
        output_write_mode=mode,
        output_write_backend=backend,
    )

    if keys is not None:
        sim.output_write_keys = keys

    sim.write_frequency = int(write_frequency)

    # Import environment rasters once.
    h5 = hdf5_io.get_hdf5_obj(sim)
    for ef in env_files:
        try:
            salmon_io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
        except Exception:
            continue

    # Prime caches and derived layers needed for compute-only mode (process backend closes HDF).
    try:
        if getattr(sim, "auto_derive_refugia", False) and not getattr(sim, "_refugia_derived", False):
            sim.derive_environment_refugia()
    except Exception:
        logger.debug("derive_environment_refugia failed during benchmark warmup", exc_info=True)
    _prime_env_cache(sim)

    # Warmup (disable outputs to avoid measuring I/O / queue effects).
    try:
        sim.disable_output_writes = True
    except Exception:
        logger.debug("Failed setting sim.disable_output_writes=True", exc_info=True)
    for i in range(int(warmup_steps)):
        sim.timestep(i, float(dt))
    try:
        sim.disable_output_writes = False
    except Exception:
        logger.debug("Failed setting sim.disable_output_writes=False", exc_info=True)

    # Timed loop: prefer calling `run()` so async backends start correctly.
    # For sync backend, this measures run loop overhead too (small).
    t0 = time.perf_counter()
    ok = sim.run(n=int(steps), dt=float(dt))
    t1 = time.perf_counter()

    try:
        sim.close()
    except Exception:
        logger.debug("sim.close failed after benchmark run", exc_info=True)

    elapsed = float(t1 - t0)
    steps_s = (float(steps) / elapsed) if elapsed > 0 else float("nan")
    agent_steps_s = (float(steps) * float(nagents) / elapsed) if elapsed > 0 else float("nan")
    return {
        "ok": bool(ok),
        "elapsed_s": elapsed,
        "steps_per_s": steps_s,
        "agent_steps_per_s": agent_steps_s,
        "db_path": db_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nagents", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--backend", choices=("sync", "thread", "process", "shm"), default="sync")
    ap.add_argument("--mode", choices=("full", "minimal", "none"), default="none")
    ap.add_argument("--write-frequency", type=int, default=0)
    ap.add_argument("--keys", type=str, default=None, help="Comma list of dataset keys to write for async backends")
    ap.add_argument("--tag", type=str, default="bench")
    ap.add_argument("--runs", type=int, default=1, help="Repeat runs and report summary stats")
    ap.add_argument("--outdir", type=str, default=os.path.join("outputs", "profiling"))
    args = ap.parse_args()

    keys = _parse_keys(args.keys)
    runs = max(1, int(args.runs))
    results = []
    for i in range(runs):
        tag_i = str(args.tag) if runs == 1 else f"{args.tag}_r{i+1}"
        res = _run_one(
            nagents=int(args.nagents),
            steps=int(args.steps),
            dt=float(args.dt),
            warmup_steps=int(args.warmup_steps),
            backend=str(args.backend),
            mode=str(args.mode),
            write_frequency=int(args.write_frequency),
            keys=keys,
            tag=tag_i,
            outdir=str(args.outdir),
        )
        results.append(res)
        print(
            f"run={i+1}/{runs} ok={res['ok']} elapsed_s={res['elapsed_s']:.3f} steps_per_s={res['steps_per_s']:.1f} "
            f"agent_steps_per_s={res['agent_steps_per_s']:.0f} db={res['db_path']}"
        )

    el = np.array([r["elapsed_s"] for r in results], dtype=float)
    ok_all = all(bool(r["ok"]) for r in results)
    if runs > 1:
        print(
            f"summary runs={runs} ok_all={ok_all} elapsed_s mean={float(np.mean(el)):.3f} p50={float(np.median(el)):.3f} "
            f"min={float(np.min(el)):.3f} max={float(np.max(el)):.3f}"
        )
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
