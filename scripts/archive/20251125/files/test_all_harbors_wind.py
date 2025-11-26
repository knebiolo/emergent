"""
Comprehensive test of wind loading for ALL harbors in config.py
Tests HRRR → ERA5 → NOAA OFS fallback chain
"""
import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP

def test_wind_harbor(port_name):
    """Test wind loading for a single harbor"""
    print(f"\n{'='*70}")
    print(f"TESTING WIND: {port_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Get bounds
        bounds = SIMULATION_BOUNDS[port_name]
        bbox = (bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'])
        
        print(f"  Bounds: lon=[{bounds['minx']:.2f}, {bounds['maxx']:.2f}], "
              f"lat=[{bounds['miny']:.2f}, {bounds['maxy']:.2f}]")
        
        # Load wind function
        print(f"  Loading wind sampler...")
        wind_fn = wind_sampler(bbox, datetime.now())
        load_time = time.time() - start_time
        
        # Sample at 3 points
        lon_min, lon_max = bounds['minx'], bounds['maxx']
        lat_min, lat_max = bounds['miny'], bounds['maxy']
        
        test_points = [
            ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, "Center"),
            (lon_min + 0.3*(lon_max - lon_min), lat_min + 0.3*(lat_max - lat_min), "SW"),
            (lon_max - 0.3*(lon_max - lon_min), lat_max - 0.3*(lat_max - lat_min), "NE")
        ]
        
        print(f"\n  Sampling at 3 locations:")
        samples = []
        now = datetime.now()
        
        for lon, lat, name in test_points:
            lons = np.array([lon])
            lats = np.array([lat])
            result = wind_fn(lons, lats, now)
            u, v = result[0, 0], result[0, 1]
            speed = np.sqrt(u**2 + v**2)
            
            samples.append((u, v, speed))
            print(f"    {name:6s} ({lon:7.2f}, {lat:6.2f}): u={u:7.3f}, v={v:7.3f}, speed={speed:.3f} m/s")
        
        # Check for spatial variation
        u_vals = [s[0] for s in samples]
        v_vals = [s[1] for s in samples]
        speeds = [s[2] for s in samples]
        
        u_range = max(u_vals) - min(u_vals)
        v_range = max(v_vals) - min(v_vals)
        speed_range = max(speeds) - min(speeds)
        
        print(f"\n  Spatial variation:")
        print(f"    U range: {u_range:.3f} m/s")
        print(f"    V range: {v_range:.3f} m/s")
        print(f"    Speed range: {speed_range:.3f} m/s")
        
        if u_range > 0.01 or v_range > 0.01:
            variation = "✓ Spatial variation confirmed!"
        else:
            variation = "⚠ Uniform wind field"
        print(f"    {variation}")
        
        total_time = time.time() - start_time
        print(f"\n  ✓ SUCCESS! Total time: {total_time:.1f}s")
        
        # Determine source
        source = "Unknown"
        if "HRRR" in str(wind_fn):
            source = "HRRR"
        elif "ERA5" in str(wind_fn):
            source = "ERA5"
        else:
            source = "NOAA OFS"
        
        return True, load_time, samples, source, None
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        print(f"\n  ✗ FAILED after {total_time:.1f}s: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, 0, None, None, error_msg


# Main test loop
print("="*70)
print("COMPREHENSIVE WIND TEST")
print("="*70)
print(f"\nTesting wind loading for {len(SIMULATION_BOUNDS)} harbors...")
print("Fallback chain: HRRR (3km CONUS) → ERA5 (31km global) → NOAA OFS")

results = {}
start_all = time.time()

for port_name in sorted(SIMULATION_BOUNDS.keys()):
    success, load_time, samples, source, error = test_wind_harbor(port_name)
    results[port_name] = {
        'success': success,
        'load_time': load_time,
        'samples': samples,
        'source': source,
        'error': error
    }

total_time = time.time() - start_all

# Summary Report
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

successful = [p for p, r in results.items() if r['success']]
failed = [p for p, r in results.items() if not r['success']]

print(f"\nTotal harbors tested: {len(results)}")
print(f"✓ Successful: {len(successful)}")
print(f"✗ Failed: {len(failed)}")
print(f"Total test time: {total_time:.1f}s")

if successful:
    print(f"\n{'='*70}")
    print("SUCCESSFUL HARBORS:")
    print(f"{'='*70}")
    print(f"{'Harbor':<25s} {'Source':<12s} {'Load Time':<12s} {'Status'}")
    print("-"*70)
    for port in sorted(successful):
        r = results[port]
        source_str = r['source'] if r['source'] else "Unknown"
        print(f"{port:<25s} {source_str:<12s} {r['load_time']:>8.1f}s     ✓")

if failed:
    print(f"\n{'='*70}")
    print("FAILED HARBORS:")
    print(f"{'='*70}")
    for port in sorted(failed):
        r = results[port]
        print(f"\n{port}:")
        print(f"  Error: {r['error']}")

# Source-specific summary
print(f"\n{'='*70}")
print("BY DATA SOURCE:")
print(f"{'='*70}")
by_source = {}
for port, r in results.items():
    if r['success']:
        source = r['source'] if r['source'] else "Unknown"
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(port)

for source in sorted(by_source.keys()):
    ports = by_source[source]
    print(f"\n{source}: {len(ports)} harbor(s)")
    for port in ports:
        print(f"  ✓ {port}")

print("\n" + "="*70)
print("\nRECOMMENDATION:")
if len(successful) == len(SIMULATION_BOUNDS):
    print("✓ All harbors have working wind data!")
    print("  Ready to run ship simulations with environmental forcing.")
else:
    print(f"⚠ {len(failed)} harbor(s) need attention.")
    print("  Check network access to AWS S3 buckets:")
    print("    - noaa-hrrr-pds (HRRR)")
    print("    - era5-pds (ERA5)")
    print("    - noaa-nos-ofs-pds (NOAA OFS)")
print("="*70)
