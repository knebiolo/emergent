from pathlib import Path
import json

from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

IN = Path('data/ship_abm/fsk_bridge.geojson')
OUT = Path('data/ship_abm/fsk_abutments.geojson')

if not IN.exists():
    print('Input bridge geojson not found at', IN)
    raise SystemExit(1)

feat = json.loads(IN.read_text())
if 'features' not in feat or not feat['features']:
    print('No features in', IN)
    raise SystemExit(1)

poly = shape(feat['features'][0]['geometry'])
# compute minimum rotated rectangle (oriented bounding box)
minrect = poly.minimum_rotated_rectangle
coords = list(minrect.exterior.coords)
# coords will have 5 points (closed); pick two opposite corners as abutments
# choose corner 0 and 2
if len(coords) >= 4:
    ab1 = coords[0]
    ab2 = coords[2]
else:
    # fallback: polygon centroid +/- direction
    c = poly.centroid
    ab1 = (c.x, c.y)
    ab2 = (c.x, c.y)

# Optionally, refine abutment by projecting onto original polygon boundary to nearest points
bound = poly.exterior
# snap to nearest boundary locations
ab1_pt = bound.interpolate(bound.project(Point(ab1)))
ab2_pt = bound.interpolate(bound.project(Point(ab2)))

out_fc = {
  "type": "FeatureCollection",
  "features": [
    {"type": "Feature", "properties": {"name": "FSK_abutment_1"}, "geometry": mapping(ab1_pt)},
    {"type": "Feature", "properties": {"name": "FSK_abutment_2"}, "geometry": mapping(ab2_pt)}
  ]
}

OUT.write_text(json.dumps(out_fc, indent=2))
print('Wrote', OUT)
print('Abutment1 (UTM):', ab1_pt.x, ab1_pt.y)
print('Abutment2 (UTM):', ab2_pt.x, ab2_pt.y)
