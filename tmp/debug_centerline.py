import io
import h5py
import numpy as np
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_arrays

# build centers and depths like the test
xs = np.linspace(0.0, 20.0, 9)
ys = np.linspace(0.0, 20.0, 9)
xv, yv = np.meshgrid(xs, ys)
centers = np.vstack([xv.ravel(), yv.ravel()]).T
depth = np.zeros((centers.shape[0],), dtype=float)
mid = centers.shape[0] // 2
depth[mid - 4: mid + 5] = 0.2
res = infer_wetted_perimeter_from_arrays(centers, depth, depth_threshold=0.05, raster_fallback_resolution=5.0)
print('result type:', type(res))
if res is None:
    print('None returned')
else:
    print('result shape:', res.shape)
    print(res)
