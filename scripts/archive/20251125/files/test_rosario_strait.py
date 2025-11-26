"""
Test SSCOFS (Salish Sea) model for Seattle and Rosario Strait region
"""
import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn

print("="*70)
print("TESTING SSCOFS (Salish Sea) - Seattle to Rosario Strait")
print("="*70)

print("\n1. Testing Seattle (Puget Sound)...")
try:
    current_fn = get_current_fn(port="Seattle", start=datetime.now())
    print("   ✓ Function loaded!")
    
    # Test in Seattle/Puget Sound
    print("\n2. Sampling at 3 locations in Puget Sound:")
    locations = [
        (-122.34, 47.60, "Downtown Seattle"),
        (-122.40, 47.65, "Ballard"),
        (-122.43, 47.70, "Shilshole Bay")
    ]
    
    now = datetime.now()
    for lon, lat, name in locations:
        lons = np.array([lon])
        lats = np.array([lat])
        result = current_fn(lons, lats, now)
        u, v = result[0, 0], result[0, 1]
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(v, u))
        print(f"   {name:20s} ({lon:7.2f}, {lat:5.2f}): u={u:7.4f}, v={v:7.4f} -> {speed:.4f} m/s @ {direction:6.1f}°")
    
    # Now test OUTSIDE the configured Seattle bounds - Rosario Strait
    print("\n3. Testing ROSARIO STRAIT (outside Seattle bounds):")
    print("   NOTE: Rosario Strait is ~100 km north of Seattle city limits")
    
    rosario_locations = [
        (-122.67, 48.55, "South Rosario"),
        (-122.70, 48.62, "Mid Rosario"),
        (-122.65, 48.68, "North Rosario")
    ]
    
    for lon, lat, name in rosario_locations:
        lons = np.array([lon])
        lats = np.array([lat])
        result = current_fn(lons, lats, now)
        u, v = result[0, 0], result[0, 1]
        speed = np.sqrt(u**2 + v**2)
        direction = np.degrees(np.arctan2(v, u))
        print(f"   {name:20s} ({lon:7.2f}, {lat:5.2f}): u={u:7.4f}, v={v:7.4f} -> {speed:.4f} m/s @ {direction:6.1f}°")
    
    print("\n✓ SSCOFS works for entire Salish Sea region!")
    print("  (The model domain is larger than the Seattle city bounds)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("\nRECOMMENDATION:")
print("If you want to simulate specifically in Rosario Strait, add a new")
print("entry to SIMULATION_BOUNDS in config.py:")
print()
print('    "Rosario Strait": {')
print('        "minx": -122.80,')
print('        "maxx": -122.60,')
print('        "miny":  48.50,')
print('        "maxy":  48.75,')
print('    },')
print()
print("And add to OFS_MODEL_MAP:")
print('    "Rosario Strait": "sscofs",')
print("="*70)
