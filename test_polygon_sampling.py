import geopandas as gpd
from shapely.geometry import Point
import numpy as np

gdf = gpd.read_file('data/salmon_abm/start_loc_river_right.shp')
print(f'CRS: {gdf.crs}')
geom = gdf.geometry.iloc[0]
minx, miny, maxx, maxy = geom.bounds
print(f'Bounds: ({minx}, {miny}, {maxx}, {maxy})')

rng = np.random.default_rng()
pts = []
attempts = 0
target = 10

while len(pts) < target and attempts < 1000:
    x = float(rng.uniform(minx, maxx))
    y = float(rng.uniform(miny, maxy))
    if geom.contains(Point(x, y)):
        pts.append((x, y))
    attempts += 1

print(f'Sampled {len(pts)} points in {attempts} attempts')
print(f'First 3 points: {pts[:3]}')
