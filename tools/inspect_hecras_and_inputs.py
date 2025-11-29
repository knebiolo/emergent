import os
import sys
import h5py
import fiona
from shapely.geometry import shape

base = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm"
plan = os.path.join(base, '20240506', 'Nuyakuk_Production_.p05.hdf')
long_shp = os.path.join(base, 'Longitudinal', 'longitudinal.shp')
start_dir = os.path.join(base, 'starting_location')

print('Plan:', plan)
print('Longitudinal shapefile:', long_shp)
print('Starting-location dir:', start_dir)

# Inspect longitudinal shapefile
try:
    with fiona.open(long_shp, 'r') as src:
        print('\nLongitudinal shapefile CRS:', src.crs)
        print('Number of features:', len(src))
        for i, feat in enumerate(src):
            geom = shape(feat['geometry'])
            print(' Feature', i, 'type', geom.geom_type, 'bounds', geom.bounds)
            if i >= 2:
                break
except Exception as e:
    print('Failed to read longitudinal shapefile:', e)

# Inspect starting-location shapefiles
try:
    for f in os.listdir(start_dir):
        if f.lower().endswith('.shp'):
            path = os.path.join(start_dir, f)
            with fiona.open(path, 'r') as src:
                print('\nStarting file:', f)
                print(' CRS:', src.crs)
                print(' Features:', len(src))
except Exception as e:
    print('Failed to read starting locations:', e)

# Inspect HDF file groups
try:
    with h5py.File(plan, 'r') as h:
        print('\nHDF top-level groups:')
        for k in h.keys():
            try:
                obj = h[k]
                print(' ', k, '->', type(obj), 'shape' if hasattr(obj, 'shape') else '', getattr(obj, 'shape', ''))
            except Exception:
                print(' ', k)
except Exception as e:
    print('Failed to open HDF:', e)
