import time
from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
p = 'data/Nuyakuk_Production_.p08.hdf'
start = time.time()
res = infer_wetted_perimeter_from_hecras(p, depth_threshold=0.05, raster_fallback_resolution=2.0, verbose=True, timestep=90)
end = time.time()
print('returned type:', type(res))
if res is None:
    print('None')
else:
    print('num rings:', len(res))
    for i,r in enumerate(res[:3]):
        print(i, r.shape)
print('took', end-start, 's')
