"""
Quick test of multiple harbors: Seattle, Rosario Strait, and Galveston
"""
import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn

def test_harbor(port_name, test_point):
    """Test a single harbor"""
    print(f"\n{'='*70}")
    print(f"TESTING: {port_name}")
    print(f"{'='*70}")
    
    start = time.time()
    
    try:
        print(f"Loading current function for {port_name}...")
        current_fn = get_current_fn(port=port_name, start=datetime.now())
        load_time = time.time() - start
        print(f"✓ Function loaded in {load_time:.1f}s")
        
        # Test sampling
        lon, lat, loc_name = test_point
        lons = np.array([lon])
        lats = np.array([lat])
        result = current_fn(lons, lats, datetime.now())
        u, v = result[0, 0], result[0, 1]
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(v, u))
        
        print(f"\nSample at {loc_name} ({lon:.2f}, {lat:.2f}):")
        print(f"  u={u:7.4f} m/s, v={v:7.4f} m/s")
        print(f"  Speed: {speed:.4f} m/s, Direction: {direction:.1f}°")
        print(f"\n✓ {port_name} WORKING!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ {port_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test each harbor
print("="*70)
print("MULTI-HARBOR TEST")
print("="*70)

results = {}

# 1. Seattle (SSCOFS - FVCOM unstructured, large mesh)
results['Seattle'] = test_harbor(
    "Seattle",
    (-122.34, 47.60, "Downtown Seattle")
)

# 2. Rosario Strait (SSCOFS - same model, different location)
results['Rosario Strait'] = test_harbor(
    "Rosario Strait",
    (-122.70, 48.62, "Mid Rosario Strait")
)

# 3. Galveston (NGOFS2 - Northern Gulf of Mexico)
results['Galveston'] = test_harbor(
    "Galveston",
    (-94.80, 29.30, "Galveston Bay")
)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for port, success in results.items():
    status = "✓ WORKING" if success else "✗ FAILED"
    print(f"{port:20s}: {status}")
print("="*70)
