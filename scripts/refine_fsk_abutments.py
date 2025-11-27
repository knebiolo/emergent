from pathlib import Path
import json
import numpy as np
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

IN = Path('data/ship_abm/fsk_bridge.geojson')
OUT = Path('data/ship_abm/fsk_abutments_refined.geojson')

if not IN.exists():
    print('Input bridge geojson not found at', IN)
    raise SystemExit(1)

feat = json.loads(IN.read_text())
poly = shape(feat['features'][0]['geometry'])
coords = np.array(poly.exterior.coords)
# PCA: center and take principal axis
center = coords.mean(axis=0)
coords_centered = coords - center
u, s, vh = np.linalg.svd(coords_centered, full_matrices=False)
principal = vh[0]
# project points onto principal axis, take min/max projections as endpoints
projs = coords_centered.dot(principal)
min_idx = np.argmin(projs)
max_idx = np.argmax(projs)
pt_min = coords[min_idx]
pt_max = coords[max_idx]
# snap to boundary by projecting
boundary = poly.exterior
pt_min_snapped = boundary.interpolate(boundary.project(Point(pt_min)))
pt_max_snapped = boundary.interpolate(boundary.project(Point(pt_max)))

fc = {"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"FSK_abutment_refined_1"},"geometry":mapping(pt_min_snapped)},
    {"type":"Feature","properties":{"name":"FSK_abutment_refined_2"},"geometry":mapping(pt_max_snapped)}
]}
OUT.write_text(json.dumps(fc, indent=2))
print('Wrote', OUT)
print('Refined abutment 1 (UTM):', pt_min_snapped.x, pt_min_snapped.y)
print('Refined abutment 2 (UTM):', pt_max_snapped.x, pt_max_snapped.y)
