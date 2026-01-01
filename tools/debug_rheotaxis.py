"""
Quick diagnostic to check rheotaxis cue values at t=0
"""
import numpy as np
import h5py
from emergent.salmon_abm import simulation

# Load latest simulation output
import glob
outputs = glob.glob('outputs/sim_db_*.h5')
if outputs:
    latest = max(outputs, key=lambda p: p)
    print(f"Loading {latest}")
    
    with h5py.File(latest, 'r') as f:
        if 'agent_data/X' in f:
            X0 = f['agent_data/X'][:, 0]  # First timestep positions
            Y0 = f['agent_data/Y'][:, 0]
            print(f"Loaded {len(X0)} agents at t=0")
            print(f"X range: {np.min(X0):.1f} to {np.max(X0):.1f}")
            print(f"Y range: {np.min(Y0):.1f} to {np.max(Y0):.1f}")
            
        # Check velocity field at starting positions
        if 'environment/vel_x' in f and 'environment/vel_y' in f:
            vel_x = f['environment/vel_x'][:]
            vel_y = f['environment/vel_y'][:]
            print(f"\nVelocity raster shape: {vel_x.shape}")
            print(f"vel_x range: {np.nanmin(vel_x):.3f} to {np.nanmax(vel_x):.3f}")
            print(f"vel_y range: {np.nanmin(vel_y):.3f} to {np.nanmax(vel_y):.3f}")
            print(f"NaN count in vel_x: {np.sum(np.isnan(vel_x))}")
            print(f"NaN count in vel_y: {np.sum(np.isnan(vel_y))}")
            
        # Check if there are headings stored
        if 'agent_data/heading' in f:
            h0 = f['agent_data/heading'][:, 0]
            print(f"\nHeading at t=0:")
            print(f"  Mean: {np.degrees(np.nanmean(h0)):.1f}°")
            print(f"  Std: {np.degrees(np.nanstd(h0)):.1f}°")
            
            # Count headings in different quadrants
            h_deg = np.degrees(h0)
            north = np.sum((h_deg > 45) & (h_deg < 135))  # 45-135° is north-ish
            east = np.sum((h_deg > -45) & (h_deg < 45))   # -45-45° is east-ish
            south = np.sum((h_deg > -135) & (h_deg < -45)) # -135--45° is south-ish
            west = np.sum((h_deg > 135) | (h_deg < -135))  # ±135-180° is west-ish
            
            print(f"\nHeading distribution:")
            print(f"  North (45-135°): {north} agents")
            print(f"  East (-45-45°): {east} agents")
            print(f"  South (-135--45°): {south} agents")
            print(f"  West (±135-180°): {west} agents")
else:
    print("No simulation outputs found")
