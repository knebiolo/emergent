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
        print("⚠️  No HDF5 object available for environment import")
        return
    
    for ef in env_files:
        try:
            arr, tr_tup, crs = io.write_raster_to_hdf5(h5, ef, dataset_name=None, sim=sim)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            print(f"✓ Imported: {key}")
        except Exception as e:
            print(f"✗ Failed to import {os.path.basename(ef)}: {e}")
    
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
                # affine.Affine object - convert to GDAL 6-tuple
                a, b, c, d, e, f = transform.to_gdal()
            elif isinstance(transform, (list, tuple)) and len(transform) == 6:
                a, b, c, d, e, f = transform
            else:
                raise ValueError(f"depth_rast_transform must be affine.Affine or 6-element tuple, got: {type(transform)}")
            
            cols = np.arange(ncols, dtype=float)
            rows = np.arange(nrows, dtype=float)
            col_indices, row_indices = np.meshgrid(cols, rows)
            x_coords = a * col_indices + b * row_indices + c
            y_coords = d * col_indices + e * row_indices + f
            hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
            hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
            print("✓ Generated coordinate grids")
    except Exception as e:
        print(f"✗ Failed to generate coordinate grids: {e}")


def run_test(args):
    """Run test simulation with diagnostics."""
    
    # Setup paths
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm'))
    base = os.path.abspath(base)
    env_files = discover_env_files(base)
    start_poly = os.path.join(base, 'start_loc_river_right.shp')
    
    outdir = os.path.abspath(os.path.join('outputs', 'test'))
    os.makedirs(outdir, exist_ok=True)
    
    print("=" * 70)
    print("SALMON ABM - CANONICAL TEST SCRIPT")
    print("=" * 70)
    print(f"Agents:    {args.nagents}")
    print(f"Steps:     {args.nsteps}")
    print(f"Seed:      {args.seed if args.seed else 'Random'}")
    print(f"Debug:     {args.debug_behavior or args.debug_movement}")
    print(f"Output:    {outdir}")
    print("=" * 70)
    
    # Create simulation
    sim = simulation(
        model_dir=outdir,
        model_name=args.model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly if os.path.exists(start_poly) else None,
        env_files=env_files,
        longitudinal_profile=None,
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=os.path.join(outdir, f'{args.model_name}.h5')
    )
    
    # Apply deterministic seed for testing
    if args.seed is not None:
        np.random.seed(args.seed)
        try:
            sim.rng = np.random.default_rng(args.seed)
        except Exception:
            sim.rng = None
        print(f"✓ Applied seed: {args.seed}")
    
    # Load test weights if provided
    if args.test_weights_file and os.path.exists(args.test_weights_file):
        try:
            with open(args.test_weights_file, 'r') as f:
                sim.test_weights = json.load(f)
            print(f"✓ Loaded test weights: {args.test_weights_file}")
        except Exception as e:
            print(f"✗ Failed to load test weights: {e}")
    
    # Enable debug modes
    if args.debug_movement:
        sim.debug_movement = True
        print("✓ Movement debugging enabled")
    if args.debug_behavior:
        sim.debug_behavior = True
        print("✓ Behavior debugging enabled")
    
    # Import environment data
    print("\nImporting environment rasters...")
    import_env_to_h5(sim, env_files)
    
    # Initialize headings from velocity field
    try:
        sim.initialize_headings_from_db()
        print("✓ Initialized headings from velocity field")
    except Exception as e:
        print(f"⚠️  Failed to initialize headings: {e}")
    
    # Validate initial sampling
    try:
        depth_vals = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth')
        valid = np.sum(np.isfinite(depth_vals) & (depth_vals != -9999.0))
        print(f"✓ Valid depth samples: {valid}/{sim.num_agents} ({100*valid/sim.num_agents:.1f}%)")
        if valid < 0.9 * sim.num_agents:
            print(f"⚠️  WARNING: {sim.num_agents - valid} agents in nodata regions!")
    except Exception as e:
        print(f"✗ Sampling validation failed: {e}")
    
    # Run simulation
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION")
    print("=" * 70)
    
    csv_path = os.path.join(outdir, f'{args.model_name}_trace.csv')
    header = ['timestep', 'agent', 'x', 'y', 'heading_deg', 'depth', 'vel_mag']
    
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        
        dt = 1.0
        start_time = time.time()
        
        for t in range(args.nsteps):
            sim.current_step = t
            step_start = time.time()
            
            sim.timestep(t, dt)
            
            # Sample environment
            depth_vals = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth')
            mag_vals = sim.sample_environment(getattr(sim, 'vel_mag_rast_transform', None), 'vel_mag')
            
            # Write trace (sample every 10 agents to keep file manageable)
            for a in range(0, sim.num_agents, max(1, sim.num_agents // 100)):
                row = [
                    t, a,
                    float(sim.X[a]), float(sim.Y[a]),
                    float(np.degrees(sim.heading[a])) if hasattr(sim, 'heading') else 0.0,
                    float(depth_vals[a]) if np.isfinite(depth_vals[a]) else '',
                    float(mag_vals[a]) if np.isfinite(mag_vals[a]) else ''
                ]
                writer.writerow(row)
            
            # Progress reporting
            if (t + 1) % max(1, args.nsteps // 10) == 0:
                elapsed = time.time() - start_time
                step_time = time.time() - step_start
                rate = (t + 1) / elapsed if elapsed > 0 else 0
                print(f"  Step {t+1:4d}/{args.nsteps}  |  {step_time*1000:.1f}ms/step  |  {rate:.1f} steps/s")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Total time:    {total_time:.2f}s")
    print(f"Steps/sec:     {args.nsteps/total_time:.2f}")
    print(f"Database:      {sim.db_path}")
    print(f"Trace:         {csv_path}")
    print("=" * 70)
    
    # Heading analysis (check for nodata bug)
    print("\nHeading Analysis (checking for nodata bug):")
    h_deg = np.degrees(sim.heading)
    north = np.sum((h_deg > 45) & (h_deg < 135))
    east = np.sum((h_deg > -45) & (h_deg < 45))
    south = np.sum((h_deg > -135) & (h_deg < -45))
    west = np.sum((h_deg > 135) | (h_deg < -135))
    
    print(f"  North (45-135°):   {north:5d} ({100*north/sim.num_agents:5.1f}%)")
    print(f"  East (-45-45°):    {east:5d} ({100*east/sim.num_agents:5.1f}%)")
    print(f"  South (-135--45°): {south:5d} ({100*south/sim.num_agents:5.1f}%)")
    print(f"  West (135-180°):   {west:5d} ({100*west/sim.num_agents:5.1f}%)")
    
    if north > 0.1 * sim.num_agents:
        print("\n⚠️  WARNING: >10% swimming north - possible nodata bug!")
    elif west > 0.5 * sim.num_agents:
        print("\n✓ PASS: Majority swimming west (upstream)")
    
    print(f"\nTo view: python -m emergent.salmon_abm.realtime_viewer {sim.db_path}")
    
    sim.close()


def main():
    parser = argparse.ArgumentParser(description='Canonical salmon ABM test script')
    parser.add_argument('--nagents', type=int, default=200, help='Number of agents (default: 200)')
    parser.add_argument('--nsteps', type=int, default=50, help='Number of timesteps (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for deterministic runs (default: 42)')
    parser.add_argument('--model-name', default='test_salmon', help='Model name for outputs')
    parser.add_argument('--debug-movement', action='store_true', help='Enable movement debug dumps')
    parser.add_argument('--debug-behavior', action='store_true', help='Enable behavior debug dumps')
    parser.add_argument('--test-weights-file', default=None, help='JSON file with test cue weights')
    
    args = parser.parse_args()
    
    try:
        run_test(args)
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
