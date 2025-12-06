from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
import h5py, time, numpy as np
from scipy.spatial import Delaunay

p = r'data/salmon_abm/20240506/Nuyakuk_Production_.p05.hdf'
print('Testing HDF:', p)
try:
    with h5py.File(p,'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
        depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0,:])
    print('Loaded cells:', len(coords), 'depth len:', len(depth))
except Exception as e:
    print('Failed to open HDF:', e)
    raise

start = time.perf_counter()
try:
    perims = infer_wetted_perimeter_from_hecras(p, depth_threshold=0.05, max_nodes=5000)
    print('Perimeter rings returned:', len(perims))
except Exception as e:
    print('Perimeter inference failed:', e)
end = time.perf_counter()
print('Perimeter inference time: {:.2f}s'.format(end-start))

wetted = depth > 0.05
pts = coords[wetted]
vals = depth[wetted]
print('Wetted count:', len(pts))
max_nodes = 5000
if len(pts) > max_nodes:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=max_nodes, replace=False)
    pts = pts[idx]; vals = vals[idx]
print('Thinned pts:', len(pts))
start = time.perf_counter()
tri = Delaunay(pts)
print('Triangles:', tri.simplices.shape[0], 'Delaunay time: {:.2f}s'.format(time.perf_counter()-start))
print('Done')
