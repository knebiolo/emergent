import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import numpy as np
from scipy.spatial import Delaunay
from emergent.salmon_abm.tin_helpers import sample_evenly
import matplotlib.pyplot as plt

# try to auto-discover HECRAS plan in data folder
hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
plan = None
for f in os.listdir(hecras_folder):
    if f.endswith('.p05.hdf'):
        plan = os.path.join(hecras_folder, f)
        break
if plan is None:
    print('HECRAS plan not found')
    sys.exit(1)

with h5py.File(plan, 'r') as hdf:
    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])

mask = depth > 0.05
pts = coords[mask]
vals = depth[mask]

max_nodes = 5000
pts, vals = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=120)

tri = Delaunay(pts)
faces = tri.simplices

out_dir = os.path.join(REPO_ROOT, 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'tin_quick.png')

fig, ax = plt.subplots(figsize=(10, 8))
ax.triplot(pts[:,0], pts[:,1], faces, linewidth=0.2)
ax.set_aspect('equal')
ax.set_title('TIN quick preview')
fig.savefig(out_path, dpi=150)
plt.close(fig)
print('Saved', out_path)
