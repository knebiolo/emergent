# -*- coding: utf-8 -*-
"""
Explore actual S3 bucket structure
"""

import sys
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src')

import fsspec
from datetime import date

print("\n" + "="*70)
print("EXPLORING ACTUAL S3 STRUCTURE")
print("="*70 + "\n")

fs = fsspec.filesystem("s3", anon=True, requester_pays=False)

# Check what's in the most recent month directory
today = date.today()
year_month = today.strftime("%Y%m")  # e.g., "202510"

buckets_to_check = ["noaa-nos-ofs-pds", "noaa-ofs-pds"]
models_to_check = ["cbofs", "ngofs2", "sscofs"]

for bucket in buckets_to_check:
    print(f"\n{'='*70}")
    print(f"BUCKET: {bucket}")
    print(f"{'='*70}")
    
    for model in models_to_check:
        print(f"\n[{model.upper()}]")
        
        # Try netcdf/YYYYMM structure
        path = f"{bucket}/{model}/netcdf/{year_month}"
        try:
            files = fs.ls(path, detail=False)
            print(f"✓ Found {len(files)} files in {path}")
            if files:
                print(f"  Sample files:")
                for f in files[:5]:
                    print(f"    {f.split('/')[-1]}")
        except Exception as e:
            print(f"✗ {path}: {str(e)[:60]}")
        
        # Try YYYY/MM structure
        path2 = f"{bucket}/{model}/netcdf/{today.year}/{today.month:02d}"
        try:
            files = fs.ls(path2, detail=False)
            print(f"✓ Found {len(files)} files in {path2}")
            if files:
                print(f"  Sample files:")
                for f in files[:5]:
                    print(f"    {f.split('/')[-1]}")
        except Exception as e:
            print(f"✗ {path2}: {str(e)[:60]}")
        
        # Try without netcdf/ prefix
        path3 = f"{bucket}/{model}/{year_month}"
        try:
            files = fs.ls(path3, detail=False)
            print(f"✓ Found {len(files)} files in {path3}")
            if files:
                print(f"  Sample files:")
                for f in files[:5]:
                    print(f"    {f.split('/')[-1]}")
        except Exception as e:
            print(f"✗ {path3}: {str(e)[:60]}")

print("\n" + "="*70)
print("DONE")
print("="*70 + "\n")
