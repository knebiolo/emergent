"""
CANONICAL TESTING SCRIPT FOR SALMON ABM

Use this script for development, debugging, and validation.
Small agent count, deterministic, with debug features enabled.

Usage:
    # Quick test (200 agents, 50 steps)
    python tools/test_salmon_abm.py
    
    # Custom parameters
    python tools/test_salmon_abm.py --nagents 500 --nsteps 100
    
    # Enable behavior debugging
    python tools/test_salmon_abm.py --debug-behavior
    
    # Test specific cue weights
    python tools/test_salmon_abm.py --test-weights-file outputs/rl_training/best_weights.json

Outputs:
    - HDF5 database: outputs/test/<model_name>.h5
    - Trace CSV: outputs/test/<model_name>_trace.csv
    - Diagnostics: outputs/test/<model_name>_diagnostics.h5 (if --debug-behavior)
"""
import os
import sys
import time
import csv
import json
import argparse
import subprocess
import logging
import numpy as np
import h5py

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io


def discover_env_files(base_dir):
    """Find standard environment raster files."""
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def ensure_env_coordinate_grids(sim):
    """Ensure `environment/x_coords` and `environment/y_coords` exist in the sim HDF5 DB."""
    h5 = hdf5_io.get_hdf5_obj(sim)
    if h5 is None:
        raise ValueError("No HDF5 object available on simulation")

    if hdf5_io.read_dataset(h5, 'environment/x_coords', default=None) is not None and hdf5_io.read_dataset(
        h5, 'environment/y_coords', default=None
    ) is not None:
        return

    depth_ds = hdf5_io.read_dataset(h5, 'environment/depth', default=None)
    if depth_ds is None:
        raise ValueError("environment/depth dataset not found; cannot create coordinate grids")

    depth_arr = np.asarray(depth_ds)
    if depth_arr.ndim != 2:
        raise ValueError(f"environment/depth must be 2D; got shape {depth_arr.shape}")
    nrows, ncols = depth_arr.shape

    transform = getattr(sim, 'depth_rast_transform', None)
    if transform is None:
        raise ValueError("depth_rast_transform not found on simulation object")
    if isinstance(transform, (list, tuple)) and len(transform) == 6:
        a, b, c, d, e, f = transform
    elif hasattr(transform, 'to_gdal'):
        a, b, c, d, e, f = transform.to_gdal()
    else:
        raise ValueError(f"depth_rast_transform must be a 6-tuple or affine.Affine; got {type(transform)}")

    cols = np.arange(ncols, dtype=float)
    rows = np.arange(nrows, dtype=float)
    col_indices, row_indices = np.meshgrid(cols, rows)

    x_coords = col_indices * a + row_indices * b + c
    y_coords = col_indices * d + row_indices * e + f
    hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
    hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)


def _normalize_timeseries_2d(arr: np.ndarray, *, n_agents: int) -> np.ndarray:
    """Return array shaped (T, N) for time-series datasets.

    HDF5 outputs sometimes store (N, T); viewer expects (T, N).
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D time-series array; got shape {arr.shape}")
    if arr.shape[0] == int(n_agents) and arr.shape[1] != int(n_agents):
        return arr.T
    return arr


def _launch_viewer(h5_path: str, *, env_depth: str | None = None, use_gl: bool = False, force_vbo: bool = False) -> None:
    cmd = [sys.executable, "-m", "emergent.salmon_abm.realtime_viewer", str(h5_path)]
    if env_depth:
        cmd.extend(["--env-depth", str(env_depth)])
    if use_gl:
        cmd.append("--use-gl")
    if force_vbo:
        cmd.append("--force-vbo")
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Test salmon ABM")
    parser.add_argument('--nagents', type=int, default=200, help='Number of agents')
    parser.add_argument('--nsteps', type=int, default=50, help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=1.0, help='Timestep size (s)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--basin', type=str, default='nuyakuk', help='Basin name')
    parser.add_argument('--water-temp', type=float, default=10.0, help='Water temperature (deg C)')
    parser.add_argument('--env-dir', type=str, default=None, help='Directory containing environment rasters (depth.tif, vel_*.tif) and start polygons')
    parser.add_argument('--start-polygon', type=str, default=None, help='Start polygon shapefile path (default: env-dir/start_loc_river_right.shp if present)')
    parser.add_argument('--longitudinal-profile', type=str, default=None, help='Optional longitudinal profile shapefile path')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory (default: outputs/test)')
    parser.add_argument('--model-name', type=str, default=None, help='Optional model name override')
    parser.add_argument('--debug-behavior', action='store_true', help='Enable behavior state tracking')
    parser.add_argument('--debug-movement', action='store_true', help='Enable movement debugging')
    parser.add_argument('--test-weights-file', type=str, default=None, help='Path to test weights JSON')
    parser.add_argument('--view', action='store_true', help='Launch realtime_viewer after the run completes')
    parser.add_argument('--open', type=str, default=None, help='Open an existing .h5/.csv in realtime_viewer and exit')
    parser.add_argument('--hecras-plan', type=str, default=None, help='Path to HECRAS plan HDF (direct mode)')
    parser.add_argument('--hecras-start-index', type=int, default=None, help='Start index into HECRAS time series (default: 30 when hecras-plan set)')
    parser.add_argument('--hecras-time-mode', type=str, default=None, help='HECRAS time mode: time, index, loop, clamp, hold (default: loop when hecras-plan set)')
    parser.add_argument('--hecras-k', type=int, default=8, help='HECRAS IDW neighbors (k)')
    parser.add_argument('--hecras-cell-size', type=float, default=None, help='Optional HECRAS grid cell size (m) for t0 rasters')
    parser.add_argument('--hecras-wetted-threshold', type=float, default=None, help='Optional depth threshold for wetted mask at t0 (m)')
    parser.add_argument('--quiet', action='store_true', help='Suppress status output (keep only progress bar if enabled)')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar during simulation')
    parser.add_argument('--output-backend', type=str, default='sync', choices=('sync', 'thread', 'process', 'shm'), help='Output backend (sync or async writer)')
    parser.add_argument('--output-keys', type=str, default=None, help='Comma-separated output keys for async backend (use "video" for agent_data/X,Y)')
    parser.add_argument('--output-every', type=int, default=None, help='Write every N steps (sync write_frequency or async write_every_steps)')
    parser.add_argument('--output-queue-max', type=int, default=None, help='Async writer queue max size')
    parser.add_argument('--output-policy', type=str, default=None, choices=('block', 'drop_oldest', 'drop_newest'), help='Async writer backpressure policy')
    parser.add_argument('--output-flush-every', type=int, default=None, help='Flush async writer every N steps (0 disables)')
    parser.add_argument('--neighbor-update-seconds', type=float, default=None, help='Rebuild neighbor graph every N seconds')
    parser.add_argument('--avoid-history-len', type=int, default=None, help='Avoid memory history length (sparse ring buffer)')
    parser.add_argument('--skip-trace', action='store_true', help='Skip CSV trace export')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip final heading analysis')
    parser.add_argument('--video-lite', action='store_true', help='Video-only mode (minimal outputs, skip analysis/trace)')
    parser.add_argument('--viewer-gl', action='store_true', help='Use OpenGL renderer in realtime_viewer')
    parser.add_argument('--viewer-force-vbo', action='store_true', help='Force VBO path in OpenGL viewer')
    args = parser.parse_args()

    quiet = bool(args.quiet)
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)

    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    if args.open:
        _launch_viewer(args.open, use_gl=bool(args.viewer_gl), force_vbo=bool(args.viewer_force_vbo))
        return

    if args.video_lite:
        args.skip_trace = True
        args.skip_analysis = True
        if args.output_backend == 'sync':
            args.output_backend = 'thread'
        if args.output_keys is None:
            args.output_keys = 'video'

    # Discover environment files (unless using HECRAS direct mode)
    if args.env_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'salmon_abm')
    else:
        base_dir = os.path.abspath(args.env_dir)

    hecras_plan = None
    hecras_start_index = None
    hecras_time_mode = None
    if args.hecras_plan:
        hecras_plan = os.path.abspath(args.hecras_plan)
        if not os.path.exists(hecras_plan):
            print(f"[ERR] HECRAS plan not found: {hecras_plan}")
            return
        hecras_start_index = 30 if args.hecras_start_index is None else int(args.hecras_start_index)
        hecras_time_mode = str(args.hecras_time_mode or "loop").strip().lower()
        log(f"[OK] HECRAS direct mode: {os.path.basename(hecras_plan)} (start_index={hecras_start_index}, mode={hecras_time_mode})")
        env_files = []
    else:
        env_files = discover_env_files(base_dir)
        if not env_files:
            print(f"[ERR] No environment files found in {base_dir}")
            return
        log(f"[OK] Found {len(env_files)} environment files")
        for ef in env_files:
            log(f"     - {os.path.basename(ef)}")
    
    # Prepare outputs
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'test') if args.outdir is None else os.path.abspath(args.outdir)
    os.makedirs(output_dir, exist_ok=True)
    model_name = args.model_name or f"test_n{args.nagents}_s{args.nsteps}_seed{args.seed}"
    h5_path = os.path.join(output_dir, f"{model_name}.h5")
    trace_path = os.path.join(output_dir, f"{model_name}_trace.csv")

    # Seed RNG for determinism
    np.random.seed(args.seed)

    # Create simulation
    log(f"\n[OK] Creating simulation: {args.nagents} agents, {args.nsteps} steps, seed={args.seed}")
    if args.start_polygon is not None and str(args.start_polygon).strip().lower() in ("none", "null", ""):
        start_poly = None
    elif args.start_polygon:
        start_poly = os.path.abspath(args.start_polygon)
    else:
        start_poly = os.path.join(base_dir, 'start_loc_river_right.shp')
        if not os.path.exists(start_poly):
            start_poly = None

    if args.longitudinal_profile is not None and str(args.longitudinal_profile).strip().lower() in ("none", "null", ""):
        longitudinal_profile = None
    elif args.longitudinal_profile:
        longitudinal_profile = os.path.abspath(args.longitudinal_profile)
    else:
        longitudinal_profile = None

    sim = simulation(
        model_dir=output_dir,
        model_name=model_name,
        crs=None,
        basin=str(args.basin),
        water_temp=float(args.water_temp),
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=longitudinal_profile,
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=h5_path,
        output_write_mode='full',
        output_write_backend=str(args.output_backend or 'sync'),
        hecras_plan_path=hecras_plan,
        hecras_start_index=hecras_start_index,
        hecras_time_mode=hecras_time_mode,
        hecras_k=int(args.hecras_k or 8),
        hecras_cell_size=args.hecras_cell_size,
        hecras_wetted_threshold=args.hecras_wetted_threshold,
    )

    def _parse_output_keys(raw):
        if raw is None:
            return None
        key_str = str(raw).strip()
        if not key_str:
            return None
        lower = key_str.lower()
        if lower in ("video", "lite", "minimal"):
            return ("agent_data/X", "agent_data/Y", "agent_data/battery", "agent_data/heading")
        if lower in ("full", "all"):
            return (
                'agent_data/X',
                'agent_data/Y',
                'agent_data/prev_X',
                'agent_data/prev_Y',
                'agent_data/ideal_sog',
                'agent_data/Hz',
                'agent_data/battery',
                'agent_data/swim_behav',
                'agent_data/heading',
                'agent_data/heading_delta',
                'agent_data/error_magnitude',
                'agent_data/pid_adjustment_magnitude',
            )
        parts = [p.strip() for p in key_str.split(",") if p.strip()]
        return tuple(parts) if parts else None

    sim.quiet = quiet
    sim.progress = bool(args.progress or quiet)
    sim.debug_behavior = bool(args.debug_behavior and not quiet)
    sim.debug_movement = bool(args.debug_movement and not quiet)
    if quiet:
        sim.debug_freq = False
        sim.debug_env = False
        sim.verbose = False
    if args.neighbor_update_seconds is not None:
        sim.neighbor_update_seconds = float(args.neighbor_update_seconds)
    if args.avoid_history_len is not None:
        sim.avoid_history_len = int(args.avoid_history_len)

    output_keys = _parse_output_keys(args.output_keys)
    if output_keys is None and str(args.output_backend or "sync").lower() != "sync":
        output_keys = ("agent_data/X", "agent_data/Y")
    if output_keys is not None:
        sim.output_write_keys = tuple(output_keys)
    if args.output_queue_max is not None:
        sim.output_write_queue_max = int(args.output_queue_max)
    if args.output_policy is not None:
        sim.output_write_policy = str(args.output_policy)
    if args.output_flush_every is not None:
        sim.output_flush_every_steps = int(args.output_flush_every)
    if args.output_every is not None:
        sim.output_write_every_steps = int(args.output_every)
        sim.write_frequency = int(args.output_every)
        sim.flush_frequency = int(args.output_every)

    if args.test_weights_file:
        sim.load_behavioral_weights(weights_path=args.test_weights_file)
        log(f"[OK] Loaded weights from {args.test_weights_file}")

    ensure_env_coordinate_grids(sim)
    
    # Initialize headings from velocity field (if present in DB)
    log(f"\n[OK] Initializing headings from velocity field...")
    sim.initialize_headings_from_db()
    log("[OK] Initialized headings")
    
    # Run simulation
    log(f"\n[OK] Running simulation...")
    t0 = time.time()
    sim.run(n=int(args.nsteps), dt=float(args.dt))
    elapsed = time.time() - t0
    if not quiet:
        print(f"[OK] Simulation complete in {elapsed:.2f}s ({args.nsteps/elapsed:.1f} steps/s)")
    
    # Close DB handle before re-opening for trace export
    sim.close()
    log(f"[OK] Saved HDF5: {h5_path}")

    # Export trace CSV from HDF5 time-series (small + deterministic)
    if not args.skip_trace:
        try:
            with h5py.File(h5_path, 'r') as h5:
                xs = _normalize_timeseries_2d(h5['agent_data/X'][:], n_agents=int(args.nagents))
                ys = _normalize_timeseries_2d(h5['agent_data/Y'][:], n_agents=int(args.nagents))
            if xs.shape != ys.shape:
                raise ValueError(f"agent_data/X shape {xs.shape} != agent_data/Y shape {ys.shape}")
            if xs.ndim != 2:
                raise ValueError(f"agent_data/X must be 2D (T,N); got shape {xs.shape}")

            with open(trace_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestep', 'agent', 'x', 'y'])
                writer.writeheader()
                T, N = xs.shape
                for t in range(T):
                    for i in range(N):
                        writer.writerow({'timestep': int(t), 'agent': int(i), 'x': float(xs[t, i]), 'y': float(ys[t, i])})
            log(f"[OK] Saved trace: {trace_path}")
        except Exception as e:
            print(f"[ERR] Failed to save trace: {e}")
    
    # Analyze final heading distribution
    if not args.skip_analysis:
        log(f"\n[OK] Analyzing final heading distribution...")
        headings_deg = (np.degrees(np.asarray(sim.heading, dtype=float)) % 360.0)
        
        # Count agents by quadrant (N/S/E/W)
        north = np.sum((headings_deg >= 315) | (headings_deg < 45))
        east = np.sum((headings_deg >= 45) & (headings_deg < 135))
        south = np.sum((headings_deg >= 135) & (headings_deg < 225))
        west = np.sum((headings_deg >= 225) & (headings_deg < 315))
        
        log(f"  North (315-45deg):   {north:4d} agents ({100*north/args.nagents:5.1f}%)")
        log(f"  East  (45-135deg):   {east:4d} agents ({100*east/args.nagents:5.1f}%)")
        log(f"  South (135-225deg):  {south:4d} agents ({100*south/args.nagents:5.1f}%)")
        log(f"  West  (225-315deg):  {west:4d} agents ({100*west/args.nagents:5.1f}%)")
        
        # Warning if too many agents swimming north (indicates nodata bug)
        if north / args.nagents > 0.10:
            log(f"\n[WARN] {100*north/args.nagents:.1f}% of agents swimming north - may indicate nodata bug!")
            log(f"       (Expected: agents should follow flow direction)")
    
    if args.view:
        log(f"\n[OK] Launching viewer: {h5_path}")
        _launch_viewer(h5_path, use_gl=bool(args.viewer_gl), force_vbo=bool(args.viewer_force_vbo))

    log(f"\n[OK] Test complete!")


if __name__ == '__main__':
    main()
