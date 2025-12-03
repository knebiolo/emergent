import sys
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import HECRASMap, compute_affine_from_hecras
import numpy as np

if len(sys.argv) < 2:
    print('usage: debug_hecras_map.py plan.hdf')
    sys.exit(1)
plan = sys.argv[1]
print('plan:', plan)
try:
    m = HECRASMap(plan, field_names=None)
    print('HECRASMap loaded')
except Exception as e:
    print('HECRASMap load error:', e)
    raise
print('coords shape:', getattr(m, 'coords', None).shape)
print('available fields:', list(m.fields.keys()))
# sample some query points: use first 10 coords
pts = m.coords[:10]
print('sample pts:', pts.shape)
try:
    mapped = m.map_idw(pts, k=8)
    for k,v in mapped.items():
        print(k, '->', v[:5])
except Exception as e:
    print('map_idw error:', e)

# compute affine and a small grid
aff = compute_affine_from_hecras(m.coords)
print('affine:', aff)
cell = abs(aff.a)
minx, maxx = m.coords[:,0].min(), m.coords[:,0].max()
miny, maxy = m.coords[:,1].min(), m.coords[:,1].max()
w = max(1, int(np.ceil((maxx-minx)/cell)))
h = max(1, int(np.ceil((maxy-miny)/cell)))
print('grid w,h:', w, h)
cols = np.arange(w)
rows = np.arange(h)
colg, rowg = np.meshgrid(cols, rows)
# convert pixel->geo using transform
xs = aff.c + aff.a * (colg + 0.0) + aff.b * (rowg + 0.0)
ys = aff.f + aff.d * (colg + 0.0) + aff.e * (rowg + 0.0)
pts2 = np.column_stack((xs.flatten(), ys.flatten()))
print('grid pts:', pts2.shape)
try:
    mapped2 = m.map_idw(pts2[:1000], k=8)
    print('mapped2 keys:', list(mapped2.keys()))
except Exception as e:
    print('map_idw grid error:', e)

print('done')
