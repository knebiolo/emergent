"""
Comprehensive test of ALL harbors in config.py
Tests each harbor to ensure data loads and spatial variation exists
"""
import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP

def test_harbor(port_name):
    """Test a single harbor - returns (success, load_time, sample_data, error_msg)"""
    print(f"\n{'='*70}")
    print(f"TESTING: {port_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Get bounds
        bounds = SIMULATION_BOUNDS[port_name]
        model = OFS_MODEL_MAP.get(port_name, "rtofs")
        
        print(f"  Model: {model.upper()}")
        print(f"  Bounds: lon=[{bounds['minx']:.2f}, {bounds['maxx']:.2f}], "
              f"lat=[{bounds['miny']:.2f}, {bounds['maxy']:.2f}]")
        
        # Load current function
        print(f"  Loading...")
        current_fn = get_current_fn(port=port_name, start=datetime.now())
        load_time = time.time() - start_time
        
        # Sample at 3 points across the domain
        lon_min, lon_max = bounds['minx'], bounds['maxx']
        lat_min, lat_max = bounds['miny'], bounds['maxy']
        
        test_points = [
            ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2, "Center"),
            (lon_min + 0.25*(lon_max - lon_min), lat_min + 0.25*(lat_max - lat_min), "SW"),
            (lon_max - 0.25*(lon_max - lon_min), lat_max - 0.25*(lat_max - lat_min), "NE")
        ]
        
        print(f"\n  Sampling at 3 locations:")
        samples = []
        now = datetime.now()
        
        for lon, lat, name in test_points:
            lons = np.array([lon])
            lats = np.array([lat])
            result = current_fn(lons, lats, now)
            u, v = result[0, 0], result[0, 1]
            speed = np.sqrt(u**2 + v**2)
            
            samples.append((u, v, speed))
            print(f"    {name:6s} ({lon:7.2f}, {lat:6.2f}): u={u:7.4f}, v={v:7.4f}, speed={speed:.4f} m/s")
        
        # Check for spatial variation
        u_vals = [s[0] for s in samples]
        v_vals = [s[1] for s in samples]
        speeds = [s[2] for s in samples]
        
        u_range = max(u_vals) - min(u_vals)
        v_range = max(v_vals) - min(v_vals)
        speed_range = max(speeds) - min(speeds)
        
        print(f"\n  Spatial variation:")
        print(f"    U range: {u_range:.4f} m/s")
        print(f"    V range: {v_range:.4f} m/s")
        print(f"    Speed range: {speed_range:.4f} m/s")
        
        if u_range > 0.001 or v_range > 0.001:
            print(f"    ✓ Spatial variation confirmed!")
        else:
            print(f"    ⚠ WARNING: Weak spatial variation (may be stagnant water)")
        
        total_time = time.time() - start_time
        print(f"\n  ✓ SUCCESS! Total time: {total_time:.1f}s")
        
        return True, load_time, samples, None
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = str(e)
        print(f"\n  ✗ FAILED after {total_time:.1f}s: {error_msg}")
        return False, 0, None, error_msg


# Main test loop
print("="*70)
print("COMPREHENSIVE HARBOR TEST")
print("="*70)
print(f"\nTesting {len(SIMULATION_BOUNDS)} harbors...")
print(f"Models available: {set(OFS_MODEL_MAP.values())}")

results = {}
start_all = time.time()

for port_name in sorted(SIMULATION_BOUNDS.keys()):
    success, load_time, samples, error = test_harbor(port_name)
    results[port_name] = {
        'success': success,
        'load_time': load_time,
        'samples': samples,
        'error': error,
        'model': OFS_MODEL_MAP.get(port_name, 'rtofs')
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
    print(f"{'Harbor':<25s} {'Model':<10s} {'Load Time':<12s} {'Status'}")
    print("-"*70)
    for port in sorted(successful):
        r = results[port]
        print(f"{port:<25s} {r['model']:<10s} {r['load_time']:>8.1f}s     ✓")

if failed:
    print(f"\n{'='*70}")
    print("FAILED HARBORS:")
    print(f"{'='*70}")
    for port in sorted(failed):
        r = results[port]
        print(f"\n{port} ({r['model']}):")
        print(f"  Error: {r['error']}")

# Model-specific summary
print(f"\n{'='*70}")
print("BY MODEL:")
print(f"{'='*70}")
by_model = {}
for port, r in results.items():
    model = r['model']
    if model not in by_model:
        by_model[model] = {'success': [], 'failed': []}
    if r['success']:
        by_model[model]['success'].append(port)
    else:
        by_model[model]['failed'].append(port)

for model in sorted(by_model.keys()):
    total = len(by_model[model]['success']) + len(by_model[model]['failed'])
    success_count = len(by_model[model]['success'])
    print(f"\n{model.upper()}: {success_count}/{total} successful")
    if by_model[model]['success']:
        for port in by_model[model]['success']:
            print(f"  ✓ {port}")
    if by_model[model]['failed']:
        for port in by_model[model]['failed']:
            print(f"  ✗ {port}")

print("\n" + "="*70)
