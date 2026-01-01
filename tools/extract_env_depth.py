"""Extract environment depth from HDF5 and save as GeoTIFF."""
import h5py
import numpy as np
import sys

def extract_depth(h5_path, output_tif):
    """Extract environment/depth from HDF5 and save as GeoTIFF."""
    try:
        from osgeo import gdal, osr
    except ImportError:
        print("ERROR: GDAL not available. Install with: conda install gdal")
        return False
    
    with h5py.File(h5_path, 'r') as f:
        if 'environment/depth' not in f:
            print(f"ERROR: environment/depth not found in {h5_path}")
            return False
        
        depth = np.array(f['environment/depth'])
        print(f"Depth shape: {depth.shape}")
        
        # Try to get x/y coordinates
        if 'environment/x_coords' in f and 'environment/y_coords' in f:
            x_coords = np.array(f['environment/x_coords'])
            y_coords = np.array(f['environment/y_coords'])
            print(f"X range: {x_coords.min():.2f} to {x_coords.max():.2f}")
            print(f"Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
            
            # Get pixel size from coordinates
            dx = (x_coords[0, 1] - x_coords[0, 0]) if x_coords.shape[1] > 1 else 1.0
            dy = (y_coords[1, 0] - y_coords[0, 0]) if y_coords.shape[0] > 1 else 1.0
            
            # Upper left corner
            x_min = x_coords[0, 0]
            y_max = y_coords[0, 0]
        else:
            print("WARNING: No coordinate arrays found, using default geotransform")
            x_min, y_max = 0.0, depth.shape[0]
            dx, dy = 1.0, -1.0
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = depth.shape
        dataset = driver.Create(output_tif, cols, rows, 1, gdal.GDT_Float32)
        
        # Set geotransform (top-left x, pixel width, 0, top-left y, 0, pixel height)
        dataset.SetGeoTransform((x_min, dx, 0, y_max, 0, dy))
        
        # Set projection (assume UTM or just write depth as-is)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32605)  # UTM Zone 5N (Alaska)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.WriteArray(depth)
        band.SetNoDataValue(-9999.0)
        band.FlushCache()
        
        dataset = None
        print(f"Saved depth to {output_tif}")
        return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_env_depth.py <input.h5> [output.tif]")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    output_tif = sys.argv[2] if len(sys.argv) > 2 else h5_path.replace('.h5', '_depth.tif')
    
    if extract_depth(h5_path, output_tif):
        print(f"\nNow run viewer with:\npython -m emergent.salmon_abm.realtime_viewer {h5_path} --env-depth {output_tif}")
    else:
        sys.exit(1)
