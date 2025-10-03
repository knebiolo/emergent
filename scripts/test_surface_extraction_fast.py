"""
FAST test to verify:
1. NOAA data loads
2. Surface layer extracted correctly
3. Spatial variation exists
WITHOUT building slow interpolator
"""
import numpy as np
import xarray as xr
import fsspec
from datetime import datetime, timedelta

def test_fast():
    print("="*70)
    print("FAST SURFACE LAYER EXTRACTION TEST")
    print("="*70)
    
    # Connect to S3
    fs = fsspec.filesystem("s3", anon=True)
    
    # Try today's CBOFS data
    date_str = datetime.now().strftime("%Y%m%d")
    y, m, d = date_str[:4], date_str[4:6], date_str[6:8]
    
    url = f"s3://noaa-nos-ofs-pds/cbofs/netcdf/{y}/{m}/{d}/cbofs.t18z.{date_str}.fields.f001.nc"
    
    print(f"\n1. Opening: {url}")
    
    try:
        ds = xr.open_dataset(
            fs.open(url),
            engine="h5netcdf",
            chunks={"time": 1}
        )
        print("   ✓ Dataset opened!")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return
    
    print("\n2. Dataset structure:")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Variables: {list(ds.data_vars.keys())[:10]}...")  # First 10
    
    # Check u/v variables
    if "u" not in ds or "v" not in ds:
        print("   ✗ No u/v variables found!")
        return
    
    u_var = ds["u"]
    v_var = ds["v"]
    
    print(f"\n3. Velocity variables:")
    print(f"   u: dims={u_var.dims}, shape={u_var.shape}, dtype={u_var.dtype}")
    print(f"   v: dims={v_var.dims}, shape={v_var.shape}, dtype={v_var.dtype}")
    
    # Extract surface layer
    print("\n4. Extracting surface layer...")
    
    # Identify vertical dimension
    vert_dims = [d for d in u_var.dims if d in ['siglay', 's_rho', 'z', 'depth', 'sigma']]
    time_dims = [d for d in u_var.dims if 'time' in d.lower()]
    
    print(f"   Vertical dims found: {vert_dims}")
    print(f"   Time dims found: {time_dims}")
    
    # Select surface
    u_surf = u_var
    v_surf = v_var
    
    if vert_dims:
        vert_dim = vert_dims[0]
        print(f"   Selecting surface layer: {vert_dim}=-1")
        u_surf = u_surf.isel({vert_dim: -1})
        v_surf = v_surf.isel({vert_dim: -1})
        
    if time_dims:
        time_dim = time_dims[0]
        print(f"   Selecting first time: {time_dim}=0")
        u_surf = u_surf.isel({time_dim: 0})
        v_surf = v_surf.isel({time_dim: 0})
    
    print(f"\n   Surface u: dims={u_surf.dims}, shape={u_surf.shape}")
    print(f"   Surface v: dims={v_surf.dims}, shape={v_surf.shape}")
    
    # Load into memory (FAST for 2D surface data)
    print("\n5. Loading surface data into memory...")
    u_arr = u_surf.values
    v_arr = v_surf.values
    
    print(f"   Loaded: u={u_arr.shape}, v={v_arr.shape}")
    
    # Statistics
    print("\n6. Statistical Analysis:")
    u_valid = u_arr[~np.isnan(u_arr)]
    v_valid = v_arr[~np.isnan(v_arr)]
    
    print(f"   U-component:")
    print(f"      Range: [{np.min(u_valid):.4f}, {np.max(u_valid):.4f}] m/s")
    print(f"      Mean: {np.mean(u_valid):.4f} m/s, Std: {np.std(u_valid):.4f} m/s")
    print(f"      Valid points: {len(u_valid):,} / {u_arr.size:,}")
    
    print(f"   V-component:")
    print(f"      Range: [{np.min(v_valid):.4f}, {np.max(v_valid):.4f}] m/s")
    print(f"      Mean: {np.mean(v_valid):.4f} m/s, Std: {np.std(v_valid):.4f} m/s")
    print(f"      Valid points: {len(v_valid):,} / {v_arr.size:,}")
    
    # Calculate speed
    speed = np.sqrt(u_arr**2 + v_arr**2)
    speed_valid = speed[~np.isnan(speed)]
    
    print(f"   Speed:")
    print(f"      Range: [{np.min(speed_valid):.4f}, {np.max(speed_valid):.4f}] m/s")
    print(f"      Mean: {np.mean(speed_valid):.4f} m/s, Std: {np.std(speed_valid):.4f} m/s")
    
    # VERDICT
    print("\n7. SPATIAL VARIATION CHECK:")
    u_range = np.max(u_valid) - np.min(u_valid)
    v_range = np.max(v_valid) - np.min(v_valid)
    speed_range = np.max(speed_valid) - np.min(speed_valid)
    
    print(f"   U variation: {u_range:.4f} m/s")
    print(f"   V variation: {v_range:.4f} m/s")
    print(f"   Speed variation: {speed_range:.4f} m/s")
    
    if u_range > 0.05 or v_range > 0.05:
        print("\n   ✓✓✓ CONFIRMED: Strong spatial variation exists! ✓✓✓")
        print(f"   Currents vary by ±{u_range/2:.3f} m/s (U) and ±{v_range/2:.3f} m/s (V)")
    elif u_range > 0.01 or v_range > 0.01:
        print("\n   ✓ CONFIRMED: Moderate spatial variation exists")
    else:
        print("\n   ✗ WARNING: Very weak spatial variation")
    
    # Sample a few random points
    print("\n8. Sample velocities at 5 random locations:")
    flat_u = u_arr.ravel()
    flat_v = v_arr.ravel()
    valid_idx = np.where(~np.isnan(flat_u) & ~np.isnan(flat_v))[0]
    
    if len(valid_idx) > 5:
        sample_idx = np.random.choice(valid_idx, 5, replace=False)
        for i, idx in enumerate(sample_idx, 1):
            u_val = flat_u[idx]
            v_val = flat_v[idx]
            spd = np.sqrt(u_val**2 + v_val**2)
            ang = np.degrees(np.arctan2(v_val, u_val))
            print(f"   Point {i}: u={u_val:7.4f}, v={v_val:7.4f} → speed={spd:.4f} m/s, dir={ang:6.1f}°")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_fast()
