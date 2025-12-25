import rasterio
import numpy as np
raster_path = 'data/salmon_abm/vel_x.tif'
with rasterio.open(raster_path) as src:
    arr = src.read(1)
    print('shape', arr.shape, 'dtype', arr.dtype)
    print('min/max', np.nanmin(arr), np.nanmax(arr))
    h, w = arr.shape
    sample_coords = [(0,0),(h//2,w//2),(h-1,w-1),(100,100),(200,300)]
    for r,c in sample_coords:
        print('pixel', (r,c), 'value', arr[r,c])
