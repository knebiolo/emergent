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


def main():
    parser = argparse.ArgumentParser(description="Test salmon ABM")
    parser.add_argument('--nagents', type=int, default=200, help='Number of agents')
    parser.add_argument('--nsteps', type=int, default=50, help='Number of simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug-behavior', action='store_true', help='Enable behavior state tracking')
    parser.add_argument('--debug-movement', action='store_true', help='Enable movement debugging')
    parser.add_argument('--test-weights-file', type=str, default=None, help='Path to test weights JSON')
    args = parser.parse_args()
    
    # Discover environment files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'salmon_abm')
    env_files = discover_env_files(base_dir)
    
    if not env_files:
        print(f"[ERR] No environment files found in {base_dir}")
        return
    
    print(f"[OK] Found {len(env_files)} environment files")
    for ef in env_files:
        print(f"     - {os.path.basename(ef)}")
    
    # Prepare outputs
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'test')
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"test_n{args.nagents}_s{args.nsteps}_seed{args.seed}"
    h5_path = os.path.join(output_dir, f"{model_name}.h5")
    trace_path = os.path.join(output_dir, f"{model_name}_trace.csv")

    # Seed RNG for determinism
    np.random.seed(args.seed)

    # Create simulation
    print(f"\n[OK] Creating simulation: {args.nagents} agents, {args.nsteps} steps, seed={args.seed}")
    start_poly = os.path.join(base_dir, 'start_loc_river_right.shp')
    if not os.path.exists(start_poly):
        start_poly = None

    sim = simulation(
        model_dir=output_dir,
        model_name=model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=None,
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=h5_path,
        output_write_mode='full',
        output_write_backend='sync',
    )

    sim.debug_behavior = bool(args.debug_behavior)
    sim.debug_movement = bool(args.debug_movement)

    if args.test_weights_file:
        sim.load_behavioral_weights(weights_path=args.test_weights_file)
        print(f"[OK] Loaded weights from {args.test_weights_file}")

    ensure_env_coordinate_grids(sim)
    
    # Initialize headings from velocity field (if present in DB)
    print(f"\n[OK] Initializing headings from velocity field...")
    sim.initialize_headings_from_db()
    print("[OK] Initialized headings")
    
    # Run simulation
    print(f"\n[OK] Running simulation...")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    print(f"[OK] Simulation complete in {elapsed:.2f}s ({args.nsteps/elapsed:.1f} steps/s)")
    
    # Close DB handle before re-opening for trace export
    sim.close()
    print(f"[OK] Saved HDF5: {h5_path}")

    # Export trace CSV from HDF5 time-series (small + deterministic)
    try:
        with h5py.File(h5_path, 'r') as h5:
            xs = np.asarray(h5['agent_data/X'])
            ys = np.asarray(h5['agent_data/Y'])
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
        print(f"[OK] Saved trace: {trace_path}")
    except Exception as e:
        print(f"[ERR] Failed to save trace: {e}")
    
    # Analyze final heading distribution
    print(f"\n[OK] Analyzing final heading distribution...")
    headings_deg = (np.degrees(np.asarray(sim.heading, dtype=float)) % 360.0)
    
    # Count agents by quadrant (N/S/E/W)
    north = np.sum((headings_deg >= 315) | (headings_deg < 45))
    east = np.sum((headings_deg >= 45) & (headings_deg < 135))
    south = np.sum((headings_deg >= 135) & (headings_deg < 225))
    west = np.sum((headings_deg >= 225) & (headings_deg < 315))
    
    print(f"  North (315-45deg):   {north:4d} agents ({100*north/args.nagents:5.1f}%)")
    print(f"  East  (45-135deg):   {east:4d} agents ({100*east/args.nagents:5.1f}%)")
    print(f"  South (135-225deg):  {south:4d} agents ({100*south/args.nagents:5.1f}%)")
    print(f"  West  (225-315deg):  {west:4d} agents ({100*west/args.nagents:5.1f}%)")
    
    # Warning if too many agents swimming north (indicates nodata bug)
    if north / args.nagents > 0.10:
        print(f"\n[WARN] {100*north/args.nagents:.1f}% of agents swimming north - may indicate nodata bug!")
        print(f"       (Expected: agents should follow flow direction)")
    
    print(f"\n[OK] Test complete!")


if __name__ == '__main__':
    main()
