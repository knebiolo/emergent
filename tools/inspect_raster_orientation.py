import rasterio
import os

raster_path = 'data/salmon_abm/vel_x.tif'
if not os.path.exists(raster_path):
    print('Raster missing:', raster_path)
    raise SystemExit(1)

with rasterio.open(raster_path) as src:
    print('CRS:', src.crs)
    print('Width, Height:', src.width, src.height)
    print('Transform:', src.transform)
    # top-left and bottom-left coordinates
    tl = src.transform * (0.5, 0.5)
    bl = src.transform * (0.5, src.height - 0.5)
    print('Top-left geo (x,y):', tl)
    print('Bottom-left geo (x,y):', bl)
    # check if y decreases with row index
    print('Row 0 Y:', (src.transform * (0.5, 0.5))[1])
    print('Row last Y:', (src.transform * (0.5, src.height - 0.5))[1])
    if (src.transform * (0.5, 0.5))[1] > (src.transform * (0.5, src.height - 0.5))[1]:
        print('Raster Y decreases with increasing row: typical geospatial origin at top-left')
    else:
        print('Raster Y increases with increasing row: likely origin at bottom-left (flipped)')

# try a roundtrip: pixel -> geo -> pixel
row, col = 10, 20
geo = src.transform * (col + 0.5, row + 0.5)
inv = ~src.transform
rt_col, rt_row = inv * (geo[0], geo[1])
print('roundtrip row,col -> geo -> row,col:', row, col, '->', geo, '->', (rt_row, rt_col))
