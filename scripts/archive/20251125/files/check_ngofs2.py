"""Check NGOFS2 (Galveston) availability and grid type"""
import fsspec
from datetime import datetime, timedelta

fs = fsspec.filesystem("s3", anon=True)

print("Checking NGOFS2 files...")
for days_ago in range(3):
    date = datetime.now() - timedelta(days=days_ago)
    y, m, d = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    prefix = f"noaa-nos-ofs-pds/ngofs2/netcdf/{y}/{m}/{d}/"
    
    try:
        files = fs.ls(prefix)
        relevant = [f for f in files if '.fields.' in f or '.2ds.' in f]
        print(f"\nNGOFS2 {date.strftime('%Y-%m-%d')}: {len(relevant)} files")
        if relevant:
            print(f"  Sample: {relevant[0].split('/')[-1]}")
    except Exception as e:
        print(f"\nNGOFS2 {date.strftime('%Y-%m-%d')}: ERROR - {e}")
