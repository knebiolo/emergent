from pyproj import Transformer

# Wikipedia coordinates: 39°12'56"N 76°31'47"W
lat = 39 + 12/60 + 56/3600
lon = -(76 + 31/60 + 47/3600)
transformer = Transformer.from_crs('EPSG:4326','EPSG:32618',always_xy=True)
x,y = transformer.transform(lon,lat)
print('Wikipedia lat,lon ->', lat, lon)
print('Converted EPSG:32618 ->', x, y)
