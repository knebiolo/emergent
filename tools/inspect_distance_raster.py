import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import h5py
import numpy as np
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters
from scipy.ndimage import distance_transform_edt

if len(sys.argv) < 2:
    print('usage: inspect_distance_raster.py plan.hdf')
    sys.exit(1)
plan = sys.argv[1]
print('plan:', plan)
h = h5py.File(plan, 'r+')
class Dummy:
    pass
sim = Dummy()
sim.hdf5 = h
sim.hecras_plan_path = plan
sim.hecras_fields = ['Cell Hydraulic Depth', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y']

print('ensuring coords...')
ensure_hdf_coords_from_hecras(sim, plan)
aff = getattr(sim, 'depth_rast_transform', None)
print('depth_rast_transform:', aff)
print('mapping...')
ok = map_hecras_to_env_rasters(sim, plan, field_names=sim.hecras_fields, k=8)
print('map returned', ok)
env = h.get('environment')
if env is None:
    print('no environment group')
    h.close(); sys.exit(2)

# prefer depth then wetted
if 'distance_to' in env:
    distance = np.array(env['distance_to'])
    print('distance found in env; shape', distance.shape)
else:
    if 'wetted' in env:
        wetted = np.array(env['wetted'])
        mask = (wetted != -9999) & (wetted > 0)
        print('wetted present; wet count', np.sum(mask))
    elif 'depth' in env:
        depth = np.array(env['depth'])
        mask = np.isfinite(depth) & (depth > 0)
        print('depth present; valid depth count', np.sum(mask), 'shape', depth.shape)
    else:
        print('no wetted/depth in env'); h.close(); sys.exit(3)
    pixel_width = aff.a if aff is not None else 1.0
    distance = distance_transform_edt(mask) * abs(pixel_width)
    print('computed distance shape', distance.shape)

# print stats
vals = distance.flatten()
finite = vals[np.isfinite(vals)]
print('stats: min', float(finite.min()), 'max', float(finite.max()), 'mean', float(finite.mean()), 'median', float(np.median(finite)))
print('nonzero count', int((finite>0).sum()), 'zero count', int((finite==0).sum()))
print('percent >1m', float((finite>1.0).sum())/finite.size)
print('percent >5m', float((finite>5.0).sum())/finite.size)

# sample max locations
mx = finite.max()
print('max distance', float(mx))

h.close()
print('done')
