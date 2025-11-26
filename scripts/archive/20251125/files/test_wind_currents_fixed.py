# -*- coding: utf-8 -*-
"""
Test script to verify wind and current data loading is working correctly.
This tests the 2D grid sampling that ships will use during simulation.

Created on October 2, 2025
@author: Kevin.Nebiolo
"""

from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def test_environmental_forcing(port="Seattle"):
    """
    Test wind and current data loading for a given port.
    Creates a 2D grid and samples environmental conditions.
    """
    print(f"\n{'='*70}")
    print(f"Testing Environmental Forcing for {port}")
    print(f"{'='*70}\n")
    
    # Get port bounds
    if port not in SIMULATION_BOUNDS:
        print(f"ERROR: Port '{port}' not found in SIMULATION_BOUNDS")
        return
    
    cfg = SIMULATION_BOUNDS[port]
    lon_min, lon_max = cfg["minx"], cfg["maxx"]
    lat_min, lat_max = cfg["miny"], cfg["maxy"]
    
    print(f"Port bounds: lon=[{lon_min:.4f}, {lon_max:.4f}], lat=[{lat_min:.4f}, {lat_max:.4f}]")
    
    # Try to get current sampler
    print(f"\n[1/2] Loading CURRENT data...")
    try:
        curr_fn = get_current_fn(port)
        print("✓ Current sampler created successfully")
    except Exception as e:
        print(f"✗ Failed to create current sampler: {e}")
        return
    
    # Try to get wind sampler
    print(f"\n[2/2] Loading WIND data...")
    try:
        wind_fn = get_wind_fn(port)
        print("✓ Wind sampler created successfully")
    except Exception as e:
        print(f"✗ Failed to create wind sampler: {e}")
        return
    
    # Create a test grid
    print(f"\n{'='*70}")
    print("Creating 10x10 test grid...")
    print(f"{'='*70}\n")
    
    n = 10
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)
    LON, LAT = np.meshgrid(lons, lats)
    
    # Sample at current time
    now = datetime.utcnow()
    print(f"Sampling at: {now.isoformat()}")
    
    # Sample currents
    print("\nSampling CURRENTS...")
    try:
        uvc = curr_fn(LON.ravel(), LAT.ravel(), now)
        curr_u = uvc[:,0].reshape(n, n)
        curr_v = uvc[:,1].reshape(n, n)
        
        print(f"✓ Current data retrieved successfully")
        print(f"  U-component range: [{np.nanmin(curr_u):.3f}, {np.nanmax(curr_u):.3f}] m/s")
        print(f"  V-component range: [{np.nanmin(curr_v):.3f}, {np.nanmax(curr_v):.3f}] m/s")
        print(f"  NaN values: {np.sum(np.isnan(curr_u))} / {n*n}")
        
        # Check for spatial variation
        if np.nanstd(curr_u) < 1e-6 and np.nanstd(curr_v) < 1e-6:
            print("  ⚠ WARNING: Current data shows no spatial variation!")
        else:
            print(f"  ✓ Spatial variation detected (std_u={np.nanstd(curr_u):.3f}, std_v={np.nanstd(curr_v):.3f})")
            
    except Exception as e:
        print(f"✗ Failed to sample currents: {e}")
        curr_u = curr_v = None
    
    # Sample winds
    print("\nSampling WINDS...")
    try:
        uvw = wind_fn(LON.ravel(), LAT.ravel(), now)
        wind_u = uvw[:,0].reshape(n, n)
        wind_v = uvw[:,1].reshape(n, n)
        
        print(f"✓ Wind data retrieved successfully")
        print(f"  U-component range: [{np.nanmin(wind_u):.3f}, {np.nanmax(wind_u):.3f}] m/s")
        print(f"  V-component range: [{np.nanmin(wind_v):.3f}, {np.nanmax(wind_v):.3f}] m/s")
        print(f"  NaN values: {np.sum(np.isnan(wind_u))} / {n*n}")
        
        # Check for spatial variation
        if np.nanstd(wind_u) < 1e-6 and np.nanstd(wind_v) < 1e-6:
            print("  ⚠ WARNING: Wind data shows no spatial variation!")
        else:
            print(f"  ✓ Spatial variation detected (std_u={np.nanstd(wind_u):.3f}, std_v={np.nanstd(wind_v):.3f})")
            
    except Exception as e:
        print(f"✗ Failed to sample winds: {e}")
        wind_u = wind_v = None
    
    # Visualization
    if curr_u is not None and wind_u is not None:
        print(f"\n{'='*70}")
        print("Creating visualization...")
        print(f"{'='*70}\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Environmental Forcing for {port}\n{now.isoformat()}', fontsize=14)
        
        # Current magnitude
        curr_mag = np.sqrt(curr_u**2 + curr_v**2)
        im0 = axes[0,0].contourf(LON, LAT, curr_mag, levels=20, cmap='viridis')
        axes[0,0].quiver(LON[::2,::2], LAT[::2,::2], 
                         curr_u[::2,::2], curr_v[::2,::2], 
                         color='white', alpha=0.7)
        axes[0,0].set_title('Current Speed (m/s)')
        axes[0,0].set_xlabel('Longitude')
        axes[0,0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0,0])
        
        # Current vectors
        axes[0,1].quiver(LON, LAT, curr_u, curr_v, curr_mag, cmap='viridis')
        axes[0,1].set_title('Current Vectors')
        axes[0,1].set_xlabel('Longitude')
        axes[0,1].set_ylabel('Latitude')
        axes[0,1].set_aspect('equal')
        
        # Wind magnitude
        wind_mag = np.sqrt(wind_u**2 + wind_v**2)
        im2 = axes[1,0].contourf(LON, LAT, wind_mag, levels=20, cmap='plasma')
        axes[1,0].quiver(LON[::2,::2], LAT[::2,::2], 
                         wind_u[::2,::2], wind_v[::2,::2], 
                         color='white', alpha=0.7)
        axes[1,0].set_title('Wind Speed (m/s)')
        axes[1,0].set_xlabel('Longitude')
        axes[1,0].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1,0])
        
        # Wind vectors
        axes[1,1].quiver(LON, LAT, wind_u, wind_v, wind_mag, cmap='plasma')
        axes[1,1].set_title('Wind Vectors')
        axes[1,1].set_xlabel('Longitude')
        axes[1,1].set_ylabel('Latitude')
        axes[1,1].set_aspect('equal')
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"environmental_forcing_{port.replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
        
        plt.show()
    
    print(f"\n{'='*70}")
    print("Test complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Test multiple ports
    ports_to_test = ["Seattle", "Galveston", "Baltimore"]
    
    for port in ports_to_test:
        try:
            test_environmental_forcing(port)
        except Exception as e:
            print(f"\n✗ Test failed for {port}: {e}\n")
        
        print("\n" + "="*70 + "\n")
