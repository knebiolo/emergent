#!/usr/bin/env python
"""Check available starting location polygons."""
import geopandas as gpd
import os

shps = [f for f in os.listdir('data/salmon_abm') if f.endswith('.shp') and 'start_loc' in f]
print('Available starting polygons:\n')

for shp in sorted(shps):
    gdf = gpd.read_file(f'data/salmon_abm/{shp}')
    geom_type = gdf.geometry.type.unique()[0]
    area = gdf.geometry.area.sum()
    
    print(f'{shp:50} Type: {geom_type:15} Area: {area:12.2f} m²')

print('\n\nRecommendation for 5000 agents:')
print('  Need at least 50,000 m² (10m²/agent) to avoid extreme density')
print('  Current options are too small - all under 30,000 m²')
print('  Solution: Use gradual spawning or accept some mortality from domain escape')
