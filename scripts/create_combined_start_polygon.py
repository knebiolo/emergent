#!/usr/bin/env python
"""Create combined starting polygon from river_left and river_right convex hull."""
import geopandas as gpd
from shapely.ops import unary_union

# Read existing polygons
left = gpd.read_file('data/salmon_abm/start_loc_river_left.shp')
right = gpd.read_file('data/salmon_abm/start_loc_river_right.shp')

# Create convex hull of all geometries
all_geoms = list(left.geometry) + list(right.geometry)
merged = unary_union(all_geoms)
convex = merged.convex_hull

# Save as new shapefile
gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[convex], crs=left.crs)
gdf.to_file('data/salmon_abm/start_loc_combined.shp')

bounds = convex.bounds
area = convex.area
print(f'Created combined polygon (CONVEX HULL):')
print(f'  Bounds: ({bounds[0]:.2f}, {bounds[1]:.2f}) to ({bounds[2]:.2f}, {bounds[3]:.2f})')
print(f'  Area: {area:.2f} m²')
print(f'  Saved to: data/salmon_abm/start_loc_combined.shp')
