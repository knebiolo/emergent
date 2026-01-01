"""
Quick test script to run 5000 agent simulation and verify rheotaxis nodata fix.
Created 2026-01-01 to test that agents no longer swim north due to -9999 velocity values.
"""
import numpy as np
from emergent.salmon_abm.simulation import simulation

print("=" * 60)
print("Testing 5000 agent simulation - Rheotaxis nodata fix")
print("=" * 60)

# Create minimal simulation (no real env files - will fail, but we can check behavior setup)
# Actually, let's use test approach instead
sim = simulation(
    model_dir='outputs',
    model_name='test_5k_rheotaxis',
    crs=None,
    basin=None,
    water_temp=np.array([10.0] * 5000),
    start_polygon=None,
    env_files=None,
    longitudinal_profile=None,
    fish_length=200.0,
    num_timesteps=50,
    num_agents=5000,
    db_path='outputs/test_5k_rheotaxis.h5',
)
sim.run()

print()
print("=" * 60)
print("Simulation complete!")
print(f"Output: {sim.db_path}")
print("=" * 60)
print()
print("Quick analysis of first movement (t0 -> t1):")

import h5py
with h5py.File(sim.db_path, 'r') as f:
    X = f['agent_data/X'][:]
    Y = f['agent_data/Y'][:]
    
    t0_x = X[:, 0]
    t0_y = Y[:, 0]
    t1_x = X[:, 1]
    t1_y = Y[:, 1]
    
    dx = t1_x - t0_x
    dy = t1_y - t0_y
    
    # Calculate headings
    h = np.arctan2(dy, dx)
    h_deg = np.degrees(h)
    
    print(f"\nMean heading: {np.mean(h_deg):.1f}°")
    print(f"Std heading: {np.std(h_deg):.1f}°")
    
    # Count by quadrant
    north = np.sum((h_deg > 45) & (h_deg < 135))
    east = np.sum((h_deg > -45) & (h_deg < 45))
    south = np.sum((h_deg > -135) & (h_deg < -45))
    west = np.sum((h_deg > 135) | (h_deg < -135))
    
    print(f"\nHeading distribution:")
    print(f"  North (45-135°):   {north:5d} agents ({100*north/len(h):5.1f}%)")
    print(f"  East (-45-45°):    {east:5d} agents ({100*east/len(h):5.1f}%)")
    print(f"  South (-135--45°): {south:5d} agents ({100*south/len(h):5.1f}%)")
    print(f"  West (135-180°):   {west:5d} agents ({100*west/len(h):5.1f}%)")
    
    print("\n" + "=" * 60)
    print("Expected: Most agents swimming WEST (upstream, ~270-300°)")
    print("Bug symptom: Many agents swimming NORTH (90°) from nodata")
    print("=" * 60)
    
    # Flag if too many agents swimming wrong direction
    if north > 0.1 * len(h):
        print("\n⚠️  WARNING: >10% of agents swimming north - possible nodata bug!")
    elif west > 0.6 * len(h):
        print("\n✅ PASS: Majority swimming west (upstream) - fix appears working!")
    else:
        print(f"\n⚠️  UNCLEAR: Headings distributed unexpectedly, check manually")

print()
print(f"To view: python -m emergent.salmon_abm.realtime_viewer {sim.db_path}")
