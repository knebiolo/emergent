"""
CANONICAL PRODUCTION SCRIPT FOR SALMON ABM

Use this script for real simulation runs with full-scale agent counts.
Optimized for performance, minimal logging, async I/O support.

Usage:
    # Standard production run
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500
    
    # With async I/O backend for better performance
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500 --backend process
    
    # Minimal writes (positions only, for live viewer)
    python tools/run_salmon_production.py --nagents 10000 --nsteps 500 --write-mode minimal

Outputs:
    - HDF5 database: outputs/production/<model_name>.h5
    - Runtime stats: outputs/production/<model_name>_stats.json
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io


def discover_env_files(base_dir):
    """Find standard environment raster files."""
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def import_env_to_h5(sim, env_files):
    """Import environment rasters into simulation HDF5."""
    h5 = hdf5_io.get_hdf5_obj(sim)
    if h5 is None:
        return
    
    for ef in env_files:
        try:
            arr, tr_tup, crs = io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
        except Exception:
            pass
    
    # Write coordinate grids
    try:
        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth')
        if depth_ds is not None:
            depth_arr = np.array(depth_ds)
            nrows, ncols = depth_arr.shape
            a, b, c, d, e, f = getattr(sim, 'depth_rast_transform', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
            cols = np.arange(ncols, dtype=float)
            rows = np.arange(nrows, dtype=float)
            col_indices, row_indices = np.meshgrid(cols, rows)
            x_coords = a * col_indices + b * row_indices + c
            y_coords = d * col_indices + e * row_indices + f
            hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
            hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
    except Exception:
        pass


def run_production(args):
    """Run production simulation."""
    
    # Setup paths
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm'))
    base = os.path.abspath(base)
    env_files = discover_env_files(base)
    start_poly = os.path.join(base, 'start_loc_river_right.shp')
    
    outdir = os.path.abspath(os.path.join('outputs', 'production'))
    os.makedirs(outdir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.model_name}_{timestamp}"
    
    print("=" * 70)
    print("SALMON ABM - PRODUCTION RUN")
    print("=" * 70)
    print(f"Model:     {model_name}")
    print(f"Agents:    {args.nagents:,}")
    print(f"Steps:     {args.nsteps:,}")
    print(f"Backend:   {args.backend}")
    print(f"Mode:      {args.write_mode}")
    print(f"Output:    {outdir}")
    print("=" * 70)
    
    # Create simulation
    db_path = os.path.join(outdir, f'{model_name}.h5')
    sim = simulation(
        model_dir=outdir,
        model_name=model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly if os.path.exists(start_poly) else None,
        env_files=env_files,
        longitudinal_profile=None,
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=db_path,
        output_write_mode=args.write_mode,
        output_write_backend=args.backend
    )
    
    # Disable debug features for performance
    sim.debug_movement = False
    sim.debug_behavior = False
    
    # Import environment data
    print("\nImporting environment...")
    import_env_to_h5(sim, env_files)
    
    # Initialize headings
    try:
        sim.initialize_headings_from_db()
        print("Initialized headings")
    except Exception:
        pass
    
    # Run simulation
    print("\n" + "=" * 70)
    print("RUNNING")
    print("=" * 70)
    
    dt = 1.0
    start_time = time.time()
    step_times = []
    
    for t in range(args.nsteps):
        step_start = time.time()
        sim.current_step = t
        sim.timestep(t, dt)
        step_times.append(time.time() - step_start)
        
        # Progress every 10%
        if (t + 1) % max(1, args.nsteps // 10) == 0:
            elapsed = time.time() - start_time
            rate = (t + 1) / elapsed
            eta = (args.nsteps - t - 1) / rate if rate > 0 else 0
            print(f"  {t+1:6d}/{args.nsteps}  |  {rate:6.1f} steps/s  |  ETA: {eta:6.1f}s")
    
    total_time = time.time() - start_time
    
    # Write runtime statistics
    stats = {
        'model_name': model_name,
        'num_agents': args.nagents,
        'num_steps': args.nsteps,
        'backend': args.backend,
        'write_mode': args.write_mode,
        'total_time_s': total_time,
        'steps_per_sec': args.nsteps / total_time,
        'mean_step_time_ms': np.mean(step_times) * 1000,
        'median_step_time_ms': np.median(step_times) * 1000,
        'std_step_time_ms': np.std(step_times) * 1000,
        'timestamp': timestamp
    }
    
    stats_path = os.path.join(outdir, f'{model_name}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total time:    {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"Steps/sec:     {stats['steps_per_sec']:.2f}")
    print(f"Step time:     {stats['mean_step_time_ms']:.1f} ± {stats['std_step_time_ms']:.1f} ms")
    print(f"Database:      {db_path}")
    print(f"Stats:         {stats_path}")
    print("=" * 70)
    print(f"\nTo view: python -m emergent.salmon_abm.realtime_viewer {db_path}")
    
    sim.close()


def main():
    parser = argparse.ArgumentParser(description='Canonical salmon ABM production script')
    parser.add_argument('--nagents', type=int, required=True, help='Number of agents')
    parser.add_argument('--nsteps', type=int, required=True, help='Number of timesteps')
    parser.add_argument('--model-name', default='salmon_prod', help='Model name prefix')
    parser.add_argument('--backend', choices=['sync', 'thread', 'process', 'shm'], 
                        default='sync', help='Output write backend (default: sync)')
    parser.add_argument('--write-mode', choices=['full', 'minimal', 'none'],
                        default='full', help='Output write mode (default: full)')
    
    args = parser.parse_args()
    
    try:
        run_production(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
