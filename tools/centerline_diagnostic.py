import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import derive_centerline_from_distance_raster
from scipy.ndimage import distance_transform_edt

if len(sys.argv) < 2:
    print('usage: centerline_diagnostic.py plan.hdf')
    sys.exit(1)
plan = sys.argv[1]
print('plan:', plan)

with h5py.File(plan, 'r') as f:
    env = f.get('environment')
    depth = None
    if env is not None and 'distance_to' in env:
        print('Using existing environment/distance_to')
        distance = np.array(env['distance_to'])
    else:
        print('No precomputed distance_to: computing from depth or wetted')
        if env is not None and 'wetted' in env:
            wetted = np.array(env['wetted'])
            mask = (wetted != -9999) & (wetted > 0)
        elif env is not None and 'depth' in env:
            depth = np.array(env['depth'])
            mask = np.isfinite(depth) & (depth > 0)
        else:
            print('No depth/wetted/distance_to found in environment; abort')
            sys.exit(2)
        # compute pixel width if possible
        transform = None
        try:
            # try to read transform from module saved var
            from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import compute_affine_from_hecras
            # derive a pixel size fallback
            pix = 1.0
        except Exception:
            pix = 1.0
        distance = distance_transform_edt(mask) * pix

print('distance shape', distance.shape)
# call helper
main, all_lines = derive_centerline_from_distance_raster(distance, transform=None, footprint_size=5, min_length=10)
print('Result main present?', bool(main))
if main is not None:
    print('Main length:', main.length)
print('Num lines returned:', len(all_lines))

# Additional introspection: compute ridge/skeleton using the same logic
from scipy.ndimage import maximum_filter
from skimage.morphology import skeletonize
from skimage.measure import label
local_max = maximum_filter(distance, size=5)
is_ridge = (distance == local_max) & (distance > 0.5)
print('ridge.sum()', int(is_ridge.sum()))
skeleton = skeletonize(is_ridge)
print('skeleton.sum()', int(skeleton.sum()))
labeled = label(skeleton, connectivity=2)
print('num components', int(labeled.max()))
# component sizes
from collections import Counter
ys, xs = np.where(skeleton)
if xs.size>0:
    comps = labeled[ys, xs]
    cnt = Counter(comps)
    # drop 0
    if 0 in cnt: del cnt[0]
    top = cnt.most_common(10)
    print('top components (id,size):', top)

print('done')
