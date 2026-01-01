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
    python tools/test_salmon_abm.py --test-weights-file scenarios/test_weights.json

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
        print("[WARN]  No HDF5 object available for environment import")
        return
    
    for ef in env_files:
        try:
            arr, tr_tup, crs = io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            print(f"[OK] Imported: {key}")
        except Exception as e:
            print(f"[ERR] Failed to import {os.path.basename(ef)}: {e}")
    
    # Write coordinate grids
    try:
        depth_ds = hdf5_io.read_dataset(h5, 'environment/depth')
        if depth_ds is not None:
            depth_arr = np.array(depth_ds)
            nrows, ncols = depth_arr.shape
            transform = getattr(sim, 'depth_rast_transform', None)
            if transform is None:
                raise ValueError("depth_rast_transform not found on simulation object")
            
            # Handle affine.Affine objects (they have 9 elements in a 3x3 matrix)
            # We need: [a, b, c, d, e, f] where x = a*col + b*row + c, y = d*col + e*row + f
            if hasattr(transform, 'to_gdal'):
                # affine - convert to GDAL 6-tuple
                a, b, c, d, e, f = transform.to_gdal()
            elif isinstance(transform, (list, tuple)) and len(transform) == 6:
                a, b, c, d, e, f = transform
            else:
                raise ValueError(f"depth_rast_transform must be affine.Affine or 6-element tuple, got: {type(transform)}")
            
            cols = np.arange(ncols, dtype=float)
            rows = np.arange(nrows, dtype=float)
            col_indices, row_indices = np.meshgrid(cols, rows)
            
            # Write coordinate grids
            hdf5_io.write_dataset(h5, 'environment/x', col_indices * a + row_indices * b + c)
            hdf5_io.write_dataset(h5, 'environment/y', col_indices * d + row_indices * e + f)
            print("[OK] Wrote coordinate grids")
    except Exception as e:
        print(f"[ERR] Failed to write coordinate grids: {e}")


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
    
    # Create simulation
    print(f"\n[OK] Creating simulation: {args.nagents} agents, {args.nsteps} steps, seed={args.seed}")
    
    config = {
        'n_agents': args.nagents,
        'n_steps': args.nsteps,
        'random_seed': args.seed,
        'dt': 1.0,
        'trace_output': True,
        'debug_behavior': args.debug_behavior,
        'debug_movement': args.debug_movement,
    }
    
    if args.test_weights_file:
        with open(args.test_weights_file, 'r') as f:
            weights = json.load(f)
            config['cue_weights'] = weights
            print(f"[OK] Loaded weights from {args.test_weights_file}")
    
    sim = simulation(**config)
    
    # Import environment data
    print(f"\n[OK] Importing environment data...")
    import_env_to_h5(sim, env_files)
    
    # Initialize agent headings from velocity field
    print(f"\n[OK] Initializing agent headings from velocity field...")
    try:
        vel_dir_ds = hdf5_io.read_dataset(hdf5_io.get_hdf5_obj(sim), 'environment/vel_dir')
        if vel_dir_ds is not None:
            vel_dir = np.array(vel_dir_ds)
            # Sample random positions and extract headings
            nrows, ncols = vel_dir.shape
            for i in range(sim.n_agents):
                row = np.random.randint(0, nrows)
                col = np.random.randint(0, ncols)
                heading = vel_dir[row, col]
                if not np.isnan(heading):
                    sim.agents[i].heading = heading
            print("[OK] Initialized headings from velocity field")
    except Exception as e:
        print(f"[WARN] Could not initialize headings: {e}")
    
    # Run simulation
    print(f"\n[OK] Running simulation...")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0
    print(f"[OK] Simulation complete in {elapsed:.2f}s ({args.nsteps/elapsed:.1f} steps/s)")
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = f"test_n{args.nagents}_s{args.nsteps}_seed{args.seed}"
    h5_path = os.path.join(output_dir, f"{model_name}.h5")
    trace_path = os.path.join(output_dir, f"{model_name}_trace.csv")
    
    # Save HDF5
    try:
        h5 = hdf5_io.get_hdf5_obj(sim)
        if h5:
            h5.close()
        print(f"[OK] Saved HDF5: {h5_path}")
    except Exception as e:
        print(f"[ERR] Failed to save HDF5: {e}")
    
    # Save trace CSV
    try:
        if hasattr(sim, 'trace_data') and sim.trace_data:
            with open(trace_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sim.trace_data[0].keys())
                writer.writeheader()
                writer.writerows(sim.trace_data)
            print(f"[OK] Saved trace: {trace_path}")
    except Exception as e:
        print(f"[ERR] Failed to save trace: {e}")
    
    # Analyze final heading distribution
    print(f"\n[OK] Analyzing final heading distribution...")
    headings = np.array([agent.heading for agent in sim.agents])
    
    # Count agents by quadrant (N/S/E/W)
    north = np.sum((headings >= 315) | (headings < 45))
    east = np.sum((headings >= 45) & (headings < 135))
    south = np.sum((headings >= 135) & (headings < 225))
    west = np.sum((headings >= 225) & (headings < 315))
    
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
