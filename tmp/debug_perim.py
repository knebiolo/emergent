import io
import h5py
import numpy as np
from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras

def make_inmemory_hecras():
    bio = io.BytesIO()
    f = h5py.File(bio, mode='w')
    grp_geom = f.create_group('Geometry/2D Flow Areas/2D area')
    xs = np.linspace(0.0, 20.0, 9)
    ys = np.linspace(0.0, 20.0, 9)
    xv, yv = np.meshgrid(xs, ys)
    centers = np.vstack([xv.ravel(), yv.ravel()]).T
    grp_geom.create_dataset('Cells Center Coordinate', data=centers)
    grp_res = f.create_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
    depth = np.zeros((1, centers.shape[0]), dtype=float)
    mid = centers.shape[0] // 2
    depth[0, mid - 4: mid + 5] = 0.2
    grp_res.create_dataset('Cell Hydraulic Depth', data=depth)
    return f

f = make_inmemory_hecras()
rings = infer_wetted_perimeter_from_hecras(f, raster_fallback_resolution=5.0)
print('rings type:', type(rings))
print('num rings:', None if rings is None else len(rings))
if rings:
    for i, r in enumerate(rings):
        print(i, 'shape', r.shape)
        print(r)
f.close()
