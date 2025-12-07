"""
Generate a TIN preview using the simulation's perimeter and save a 2D PNG (no GUI required).
This avoids opening a window while still validating that mesh clipping works.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from time import sleep

try:
    from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation
    from emergent.salmon_abm.tin_helpers import sample_evenly
except Exception as e:
    print('Import error:', e)
    raise

model_dir = os.path.abspath('.')
hecras_default = os.path.join(model_dir, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
if not os.path.exists(hecras_default):
    print('HECRAS plan not found at', hecras_default)
    sys.exit(1)

# Create a simple start polygon (convex hull of HECRAS cell centers) and write to GeoJSON
import h5py
with h5py.File(hecras_default, 'r') as hdf:
    coords_all = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
    # determine middle timestep from HECRAS depths if available
    try:
        depth_ds = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
        num_ts = int(depth_ds.shape[0])
        middle_ti = num_ts // 2
    except Exception:
        num_ts = None
        middle_ti = 0

try:
    from shapely.geometry import MultiPoint, mapping
    hull = MultiPoint([tuple(p) for p in coords_all]).convex_hull
    start_poly_path = os.path.join(model_dir, 'data', 'start_polygon.geojson')
    os.makedirs(os.path.dirname(start_poly_path), exist_ok=True)
    import json
    geojson = mapping(hull)
    with open(start_poly_path, 'w') as fh:
        json.dump({'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': geojson, 'properties': {}}]}, fh)
except Exception:
    start_poly_path = None

# instantiate simulation - supply required positional args with safe defaults
# Use the HECRAS file's timestep count if available so perimeter/TIN use the same index
sim_num_ts = num_ts if num_ts is not None else 181
sim = simulation(model_dir, 'test_model', 'EPSG:4326', 'Nushagak River', 10.0, start_poly_path, None,
                 env_files=None, fish_length=None, num_timesteps=sim_num_ts, num_agents=1,
                 use_gpu=False, pid_tuning=False, hecras_plan_path=hecras_default,
                 hecras_fields=None, hecras_k=8, use_hecras=True, hecras_write_rasters=False)

# read HECRAS file to sample points (same logic as viewer.setup_background)
import h5py
with h5py.File(hecras_default, 'r') as hdf:
    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
    depth_ds = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth']
    # use middle timestep (or 0 fallback)
    ti = middle_ti if 'middle_ti' in globals() else 0
    depth = np.array(depth_ds[ti, :])

max_nodes = getattr(sim, 'tin_max_nodes', 5000)
depth_thresh = getattr(sim, 'tin_depth_thresh', 1e-5)
mask = depth > depth_thresh
pts = coords[mask]
vals = depth[mask]
if len(pts) > max_nodes:
    pts, vals = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=120)

if len(pts) < 3:
    print('Not enough points for TIN')
    sys.exit(1)

# triangulate
from scipy.spatial import Delaunay
tri = Delaunay(pts)
faces = tri.simplices
verts = np.column_stack([pts[:,0], pts[:,1], vals])

# Clip faces using sim.perimeter_polygon if available
poly = getattr(sim, 'perimeter_polygon', None)
if poly is not None:
    try:
        from shapely.geometry import Point
        face_centroids = np.mean(verts[faces], axis=1)[:, :2]
        inside_mask = []
        for c in face_centroids:
            try:
                inside_mask.append(poly.contains(Point(float(c[0]), float(c[1]))))
            except Exception:
                inside_mask.append(False)
        inside_mask = np.array(inside_mask, dtype=bool)
        if inside_mask.sum() > 0:
            faces = faces[inside_mask]
    except Exception:
        pass

# save 2D preview
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
ax.triplot(verts[:,0], verts[:,1], faces, linewidth=0.2)
ax.set_aspect('equal')
ax.set_title('TIN preview (clipped)')
out_dir = os.path.join(model_dir, 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'tin_clipped_preview.png')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print('Saved preview to', out_path)
