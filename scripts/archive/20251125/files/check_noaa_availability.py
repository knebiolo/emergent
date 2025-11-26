# Check what NOAA actually has for today
import fsspec
from datetime import date, timedelta

fs = fsspec.filesystem("s3", anon=True)

today = date.today()
print(f"\nChecking NOAA data availability for {today}\n")

models = ["cbofs", "ngofs2", "sscofs"]
bucket = "noaa-nos-ofs-pds"

for model in models:
    print(f"\n{model.upper()}:")
    for days_ago in range(5):  # Check last 5 days
        check_date = today - timedelta(days=days_ago)
        y, m, d = check_date.strftime("%Y"), check_date.strftime("%m"), check_date.strftime("%d")
        path = f"{bucket}/{model}/netcdf/{y}/{m}/{d}"
        
        try:
            files = fs.ls(path, detail=False)
            # Filter to just .nc files
            nc_files = [f for f in files if f.endswith('.nc')]
            print(f"  {check_date} ({days_ago}d ago): {len(nc_files)} files")
            if nc_files:
                # Show sample
                sample = nc_files[0].split('/')[-1]
                print(f"    Sample: {sample}")
        except:
            print(f"  {check_date} ({days_ago}d ago): NO DATA")
