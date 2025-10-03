"""Check SSCOFS grid type"""
import xarray as xr
import fsspec

fs = fsspec.filesystem("s3", anon=True)
url = "s3://noaa-nos-ofs-pds/sscofs/netcdf/2025/10/02/sscofs.t03z.20251002.fields.f000.nc"

print(f"Opening: {url}")
# Drop problematic coordinate variables for FVCOM
ds = xr.open_dataset(fs.open(url), engine="h5netcdf", drop_variables=['siglay', 'siglev'])

print("\nDimensions:")
for k, v in ds.dims.items():
    print(f"  {k}: {v}")

print("\nCoordinate variables:")
coord_vars = [v for v in ds.variables if 'lon' in v.lower() or 'lat' in v.lower()]
for v in coord_vars:
    print(f"  {v}: shape={ds[v].shape}, dims={ds[v].dims}")

print("\nVelocity variables:")
if 'u' in ds:
    print(f"  u: shape={ds['u'].shape}, dims={ds['u'].dims}")
if 'v' in ds:
    print(f"  v: shape={ds['v'].shape}, dims={ds['v'].dims}")

# Determine grid type
if 'lon_rho' in ds or 'lon_u' in ds:
    print("\n✓ ROMS curvilinear grid detected")
elif 'lon' in ds and ds['lon'].ndim == 1:
    print("\n✓ FVCOM unstructured (triangular) grid detected")
elif 'lon' in ds and ds['lon'].ndim == 2:
    print("\n✓ FVCOM structured (rectangular) grid detected")
else:
    print("\n? Unknown grid type")
