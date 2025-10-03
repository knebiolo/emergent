# -*- coding: utf-8 -*-
"""
Simple diagnostic test - just try to connect and see what happens
"""

import sys
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src')

from datetime import datetime, date, timedelta
import numpy as np

print("\n" + "="*70)
print("SIMPLE OFS CONNECTION TEST")
print("="*70 + "\n")

# Test 1: Can we access S3?
print("[TEST 1] Checking S3 filesystem access...")
try:
    import fsspec
    fs = fsspec.filesystem("s3", anon=True, requester_pays=False)
    print("✓ fsspec loaded, filesystem created")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Can we list a known bucket?
print("\n[TEST 2] Checking bucket access...")
try:
    # Try to check if a known recent file exists
    bucket = "noaa-nos-ofs-pds"
    test_path = f"{bucket}/cbofs"
    exists = fs.exists(test_path)
    print(f"✓ Bucket accessible: {bucket}")
    print(f"  Path exists: {test_path} = {exists}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Try to find ONE actual file
print("\n[TEST 3] Searching for actual OFS files...")
try:
    from emergent.ship_abm.ofs_loader import regional_keys, BUCKETS
    
    model = "cbofs"  # Chesapeake Bay
    today = date.today()
    
    keys = regional_keys(model, today)
    print(f"✓ Generated {len(keys)} key patterns for {model} on {today}")
    print(f"  First few keys:")
    for k in keys[:3]:
        print(f"    {k}")
    
    # Now try to find which ones actually exist
    print(f"\n  Checking which files exist on S3...")
    found = []
    for key in keys[:6]:  # Check first 6
        for bucket in BUCKETS:
            full_path = f"{bucket}/{key}"
            if fs.exists(full_path):
                found.append(full_path)
                print(f"  ✓ FOUND: s3://{full_path}")
                break
    
    if found:
        print(f"\n✓ Found {len(found)} file(s)!")
    else:
        print(f"\n⚠ No files found in first 6 attempts")
        print(f"  This might be normal - let's try yesterday...")
        
        yesterday = today - timedelta(days=1)
        keys_y = regional_keys(model, yesterday)
        for key in keys_y[:6]:
            for bucket in BUCKETS:
                full_path = f"{bucket}/{key}"
                if fs.exists(full_path):
                    found.append(full_path)
                    print(f"  ✓ FOUND: s3://{full_path}")
                    break
        
        if not found:
            print(f"  ✗ Still nothing found. Checking what's in the bucket...")
            try:
                # List what's actually in there
                files = fs.ls(f"{BUCKETS[0]}/{model}/netcdf", detail=False)
                print(f"  Found {len(files)} items in {BUCKETS[0]}/{model}/netcdf/")
                if files:
                    print(f"  First few:")
                    for f in files[:5]:
                        print(f"    {f}")
            except Exception as e:
                print(f"  Could not list directory: {e}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: If we found a file, try to open it
if found:
    print(f"\n[TEST 4] Attempting to open file...")
    try:
        import xarray as xr
        
        url = f"s3://{found[0]}"
        print(f"  Opening: {url}")
        
        ds = xr.open_dataset(
            fs.open(found[0]),
            engine="h5netcdf",
            chunks={"time": 1}
        )
        
        print(f"✓ Dataset opened successfully!")
        print(f"\n  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Coordinates: {list(ds.coords)}")
        
        # Check for current variables
        print(f"\n  Looking for current variables...")
        var_pairs = [("ua", "va"), ("us", "vs"), ("u", "v"), ("water_u", "water_v")]
        found_vars = None
        for u_var, v_var in var_pairs:
            if u_var in ds and v_var in ds:
                found_vars = (u_var, v_var)
                print(f"  ✓ Found: {u_var}, {v_var}")
                
                # Get shape info
                u_shape = ds[u_var].shape
                print(f"    Shape: {u_shape}")
                print(f"    Dims: {ds[u_var].dims}")
                break
        
        if not found_vars:
            print(f"  ✗ No recognized current variables found")
            print(f"    Available: {list(ds.data_vars)}")
        
    except Exception as e:
        print(f"✗ Failed to open dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n[TEST 4] SKIPPED - no files found")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
