"""Quick check: What SSCOFS files exist on S3?"""
import fsspec
from datetime import datetime, timedelta

fs = fsspec.filesystem("s3", anon=True)

print("Checking NOAA S3 for SSCOFS files...")
print()

for days_ago in range(5):
    date = datetime.now() - timedelta(days=days_ago)
    y, m, d = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    ymd = date.strftime("%Y%m%d")
    
    prefix = f"noaa-nos-ofs-pds/sscofs/netcdf/{y}/{m}/{d}/"
    
    try:
        files = fs.ls(prefix)
        # Filter for fields files
        fields_files = [f for f in files if '.fields.f' in f or '.2ds.f' in f]
        print(f"SSCOFS: {date.strftime('%Y-%m-%d')} ({days_ago}d ago): {len(fields_files)} files")
        if fields_files:
            print(f"  Sample: {fields_files[0].split('/')[-1]}")
    except Exception as e:
        print(f"SSCOFS: {date.strftime('%Y-%m-%d')} ({days_ago}d ago): ERROR - {e}")
    print()
