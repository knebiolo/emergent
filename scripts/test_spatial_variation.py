"""
Simple test to prove spatial variation exists in NOAA ocean currents
Should complete in < 30 seconds
"""
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")
from emergent.ship_abm.ofs_loader import get_current_fn

def test_spatial_variation():
    """Load CBOFS data and sample at different locations to prove spatial variation"""
    print("="*60)
    print("TESTING SPATIAL VARIATION IN NOAA OCEAN CURRENTS")
    print("="*60)
    
    print("\n1. Loading ocean current data for Baltimore (CBOFS model)...")
    
    try:
        # get_current_fn uses 'port' name which maps to model via OFS_MODEL_MAP
        # Baltimore -> CBOFS (Chesapeake Bay)
        current_fn = get_current_fn(
            port="Baltimore",
            start=datetime.now()
        )
        print("   ✓ Data loaded successfully!")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2. Sampling at 5 different locations in Chesapeake Bay:")
    print("   Format: (lon, lat) → (u_current, v_current)")
    print()
    
    # Sample at 5 different locations within Baltimore simulation bounds
    # SIMULATION_BOUNDS["Baltimore"] = minx=-76.60, maxx=-76.30, miny=39.19, maxy=39.50
    locations = [
        (-76.60, 39.20, "SW Corner"),
        (-76.45, 39.27, "South Center"),
        (-76.45, 39.35, "Mid Center"),
        (-76.45, 39.42, "North Center"),
        (-76.30, 39.48, "NE Corner")
    ]
    
    now = datetime.now()
    velocities = []
    for lon, lat, name in locations:
        # current_fn expects (lons_array, lats_array, when) -> (N, 2) array
        lons = np.array([lon])
        lats = np.array([lat])
        result = current_fn(lons, lats, now)
        u, v = result[0, 0], result[0, 1]
        
        speed = np.sqrt(u**2 + v**2)
        angle = np.degrees(np.arctan2(v, u))
        
        velocities.append((u, v, speed))
        
        print(f"   {name:15s} ({lon:6.2f}, {lat:5.2f})")
        print(f"      → U={u:7.4f} m/s, V={v:7.4f} m/s")
        print(f"      → Speed={speed:7.4f} m/s, Direction={angle:6.1f}°")
        print()
    
    # Calculate statistics
    u_vals = [v[0] for v in velocities]
    v_vals = [v[1] for v in velocities]
    speeds = [v[2] for v in velocities]
    
    print("\n3. Statistical Analysis:")
    print(f"   U-component range: {min(u_vals):.4f} to {max(u_vals):.4f} m/s (Δ={max(u_vals)-min(u_vals):.4f})")
    print(f"   V-component range: {min(v_vals):.4f} to {max(v_vals):.4f} m/s (Δ={max(v_vals)-min(v_vals):.4f})")
    print(f"   Speed range: {min(speeds):.4f} to {max(speeds):.4f} m/s (Δ={max(speeds)-min(speeds):.4f})")
    
    # Check for variation
    u_variation = max(u_vals) - min(u_vals)
    v_variation = max(v_vals) - min(v_vals)
    
    print("\n4. VERDICT:")
    if u_variation > 0.01 or v_variation > 0.01:
        print("   ✓✓✓ SPATIAL VARIATION CONFIRMED! ✓✓✓")
        print(f"   The currents differ by {u_variation:.4f} m/s (U) and {v_variation:.4f} m/s (V)")
        print("   across different locations in the bay.")
    else:
        print("   ✗✗✗ WARNING: No spatial variation detected")
        print("   All sampled points return identical values - possible interpolation issue")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_spatial_variation()
