# -*- coding: utf-8 -*-
"""
Test that we can actually SAMPLE current data from the real OFS file
"""

import sys
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src')

print("\n" + "="*70)
print("TESTING ACTUAL DATA SAMPLING")
print("="*70 + "\n")

# Use the fixed get_current_fn
from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np

port = "Baltimore"  # uses cbofs
print(f"[1] Creating current sampler for {port}...")

try:
    curr_fn = get_current_fn(port)
    print("✓ Sampler created successfully\n")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test sampling at a grid of points
print("[2] Creating test grid...")
lons = np.array([-76.5, -76.4, -76.3])
lats = np.array([39.2, 39.3, 39.4])
LON, LAT = np.meshgrid(lons, lats)
print(f"  Grid shape: {LON.shape}")
print(f"  Lon range: [{LON.min():.2f}, {LON.max():.2f}]")
print(f"  Lat range: [{LAT.min():.2f}, {LAT.max():.2f}]\n")

# Sample
print("[3] Sampling current data...")
now = datetime.utcnow()
print(f"  Time: {now.isoformat()}")

try:
    uv = curr_fn(LON.ravel(), LAT.ravel(), now)
    print(f"✓ Data sampled successfully!")
    print(f"  Result shape: {uv.shape}")
    print(f"  Result type: {type(uv)}\n")
    
    # Reshape and analyze
    u = uv[:, 0].reshape(LON.shape)
    v = uv[:, 1].reshape(LON.shape)
    
    print("[4] Analyzing results...")
    print(f"  U-component (m/s):")
    print(f"    min={np.nanmin(u):.4f}, max={np.nanmax(u):.4f}, mean={np.nanmean(u):.4f}")
    print(f"    std={np.nanstd(u):.4f}")
    print(f"    NaN count: {np.sum(np.isnan(u))} / {u.size}")
    print(f"\n  V-component (m/s):")
    print(f"    min={np.nanmin(v):.4f}, max={np.nanmax(v):.4f}, mean={np.nanmean(v):.4f}")
    print(f"    std={np.nanstd(v):.4f}")
    print(f"    NaN count: {np.sum(np.isnan(v))} / {v.size}")
    
    # Check for spatial variation
    print(f"\n[5] Checking spatial variation...")
    if np.nanstd(u) < 1e-6 and np.nanstd(v) < 1e-6:
        print("  ⚠ WARNING: No spatial variation detected!")
    else:
        print(f"  ✓ Spatial variation confirmed!")
        print(f"    ΔU = {np.nanmax(u) - np.nanmin(u):.4f} m/s")
        print(f"    ΔV = {np.nanmax(v) - np.nanmin(v):.4f} m/s")
    
    print(f"\n  Sample values:")
    print(f"    Grid:")
    for i in range(LON.shape[0]):
        for j in range(LON.shape[1]):
            print(f"      ({LON[i,j]:.2f}, {LAT[i,j]:.2f}): u={u[i,j]:.3f}, v={v[i,j]:.3f} m/s")
    
except Exception as e:
    print(f"✗ Sampling failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("SUCCESS! Real 2D ocean current data is loading and working!")
print("="*70 + "\n")
