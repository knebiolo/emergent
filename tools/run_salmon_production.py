"""
CANONICAL PRODUCTION SCRIPT FOR SALMON ABM

Use this script for real simulation runs with full-scale agent counts.
This runner uses the current `simulation.run(n=..., dt=...)` API and relies on
`simulation(...)` to import environment rasters from `--env-dir`.

Usage:
    # Standard production run
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500
    
    # With async I/O backend for better performance
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500 --backend process
    
    # Minimal writes (positions only, for live viewer)
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500 --write-mode minimal

    # Run then launch playback viewer
    python tools/run_salmon_production.py --nagents 2000 --nsteps 200 --view

Outputs:
    - HDF5 database: outputs/production/<model_name>.h5
    - Runtime stats: outputs/production/<model_name>_stats.json
"""
import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime

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


def ensure_env_coordinate_grids(sim: simulation) -> None:
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


def _launch_viewer(h5_path: str, *, env_depth: str | None = None) -> None:
    cmd = [sys.executable, "-m", "emergent.salmon_abm.realtime_viewer", str(h5_path)]
    if env_depth:
        cmd.extend(["--env-depth", str(env_depth)])
    subprocess.run(cmd, check=False)


def run_production(args):
    """Run production simulation."""
    
    default_env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm'))
    base = os.path.abspath(args.env_dir) if getattr(args, 'env_dir', None) else default_env_dir
    env_files = discover_env_files(base)
    if not env_files:
        raise FileNotFoundError(f"No environment files found in {base} (expected depth.tif, vel_*.tif)")

    if args.start_polygon is not None and str(args.start_polygon).strip().lower() in ("none", "null", ""):
        start_poly = None
    elif args.start_polygon:
        start_poly = os.path.abspath(args.start_polygon)
    else:
        start_poly = os.path.join(base, 'start_loc_river_right.shp')
        if not os.path.exists(start_poly):
            start_poly = None

    if args.longitudinal_profile is not None and str(args.longitudinal_profile).strip().lower() in ("none", "null", ""):
        longitudinal_profile = None
    elif args.longitudinal_profile:
        longitudinal_profile = os.path.abspath(args.longitudinal_profile)
    else:
        longitudinal_profile = os.path.join(base, 'longitudinal.shp')
        if not os.path.exists(longitudinal_profile):
            longitudinal_profile = None

    outdir = os.path.abspath(args.outdir) if getattr(args, 'outdir', None) else os.path.abspath(os.path.join('outputs', 'production'))
    os.makedirs(outdir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.model_name}_{timestamp}"
    
    print("=" * 70)
    print("SALMON ABM - PRODUCTION RUN")
    print("=" * 70)
    print(f"Model:     {model_name}")
    print(f"Agents:    {args.nagents:,}")
    print(f"Steps:     {args.nsteps:,}")
    print(f"dt:        {args.dt}")
    print(f"Backend:   {args.backend}")
    print(f"Mode:      {args.write_mode}")
    print(f"Env dir:   {base}")
    print(f"Start:     {start_poly if start_poly else '(none)'}")
    print(f"Output:    {outdir}")
    print("=" * 70)
    
    # Create simulation
    db_path = os.path.join(outdir, f'{model_name}.h5')
    sim = simulation(
        model_dir=outdir,
        model_name=model_name,
        crs=None,
        basin=str(args.basin),
        water_temp=float(args.water_temp),
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=longitudinal_profile,
        num_timesteps=int(args.nsteps),
        num_agents=int(args.nagents),
        db_path=db_path,
        output_write_mode=str(args.write_mode),
        output_write_backend=str(args.backend),
    )
    
    # Disable debug features for performance
    sim.debug_movement = False
    sim.debug_behavior = False

    if getattr(args, 'weights_file', None):
        sim.load_behavioral_weights(weights_path=str(args.weights_file))
        print(f"Loaded behavioral weights: {args.weights_file}")

    ensure_env_coordinate_grids(sim)

    if not getattr(args, 'skip_init_headings', False):
        sim.initialize_headings_from_db()

    print("\n" + "=" * 70)
    print("RUNNING")
    print("=" * 70)

    start_time = time.time()
    status = None
    try:
        status = sim.run(n=int(args.nsteps), dt=float(args.dt), return_status=True)
    finally:
        sim.close()

    total_time = time.time() - start_time

    stats = {
        'model_name': model_name,
        'num_agents': int(args.nagents),
        'num_steps': int(args.nsteps),
        'dt': float(args.dt),
        'backend': str(args.backend),
        'write_mode': str(args.write_mode),
        'total_time_s': float(total_time),
        'steps_per_sec': (float(args.nsteps) / total_time) if total_time > 0 else float('nan'),
        'timestamp': timestamp,
        'db_path': db_path,
        'status': status,
    }

    stats_path = os.path.join(outdir, f'{model_name}_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time:    {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"Steps/sec:     {stats['steps_per_sec']:.2f}")
    print(f"Database:      {db_path}")
    print(f"Stats:         {stats_path}")
    print("=" * 70)
    print(f"\nTo view: python -m emergent.salmon_abm.realtime_viewer {db_path}")

    if getattr(args, 'view', False):
        _launch_viewer(db_path)


def main():
    parser = argparse.ArgumentParser(description='Canonical salmon ABM production script')
    parser.add_argument('--nagents', type=int, required=True, help='Number of agents')
    parser.add_argument('--nsteps', type=int, required=True, help='Number of timesteps')
    parser.add_argument('--model-name', default='salmon_prod', help='Model name prefix')
    parser.add_argument('--dt', type=float, default=1.0, help='Timestep size (s)')
    parser.add_argument('--basin', type=str, default='nuyakuk', help='Basin name')
    parser.add_argument('--water-temp', type=float, default=10.0, help='Water temperature (deg C)')
    parser.add_argument('--env-dir', type=str, default=None, help='Directory containing environment rasters (depth.tif, vel_*.tif) and start polygons')
    parser.add_argument('--start-polygon', type=str, default=None, help='Start polygon shapefile path (default: env-dir/start_loc_river_right.shp if present)')
    parser.add_argument('--longitudinal-profile', type=str, default=None, help='Optional longitudinal profile shapefile path (default: env-dir/longitudinal.shp if present)')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory (default: outputs/production)')
    parser.add_argument('--weights-file', type=str, default=None, help='Optional behavioral weights JSON to load (default: use source-code defaults)')
    parser.add_argument('--skip-init-headings', action='store_true', help='Skip initialize_headings_from_db()')
    parser.add_argument('--backend', choices=['sync', 'thread', 'process', 'shm'], 
                        default='sync', help='Output write backend (default: sync)')
    parser.add_argument('--write-mode', choices=['full', 'minimal', 'none'],
                        default='full', help='Output write mode (default: full)')
    parser.add_argument('--view', action='store_true', help='Launch realtime_viewer after the run completes')
    
    args = parser.parse_args()
    
    try:
        run_production(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
