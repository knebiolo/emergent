"""inspect_dry_region.py
Extracts the largest dry segment from the diagnostic binning and writes CSV + zoomed scatter + histogram.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
if not os.path.exists(data_dir):
    print('HECRAS data dir not found:', data_dir); sys.exit(1)

hdf_path = None
for f in os.listdir(data_dir):
    if f.endswith('.p05.hdf'):
        hdf_path = os.path.join(data_dir, f)
        break
if hdf_path is None:
    print('HECRAS HDF not found'); sys.exit(1)

with h5py.File(hdf_path, 'r') as hdf:
    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
coords2 = coords[:, :2]

# Reuse diagnostic binning (PCA)
mean = coords2.mean(axis=0)
X = coords2 - mean
u, s, vt = np.linalg.svd(X, full_matrices=False)
pc1 = vt[0]
proj = X.dot(pc1)

n_bins = 300
bins = np.linspace(proj.min(), proj.max(), n_bins+1)
bin_idx = np.digitize(proj, bins) - 1

# find the largest dry segment from the earlier diagnostic logic
depth_thresh = 0.05
wetted = depth > depth_thresh
frac_wetted = np.zeros(n_bins)
for i in range(n_bins):
    mask = bin_idx == i
    if mask.sum() == 0:
        frac_wetted[i] = np.nan
    else:
        frac_wetted[i] = np.sum(wetted[mask]) / mask.sum()
thr = 0.2
is_dry_bin = np.where(np.nan_to_num(frac_wetted, nan=0.0) < thr, 1, 0)
segments = []
start = None
for i, v in enumerate(is_dry_bin):
    if v == 1 and start is None:
        start = i
    if v == 0 and start is not None:
        segments.append((start, i-1)); start = None
if start is not None:
    segments.append((start, n_bins-1))
segments_sorted = sorted(segments, key=lambda s: s[1]-s[0], reverse=True)
if len(segments_sorted) == 0:
    print('No dry segments found under threshold'); sys.exit(0)

largest_seg = segments_sorted[0]
s,e = largest_seg
mask_seg = (bin_idx >= s) & (bin_idx <= e)

out_dir = os.path.join(REPO_ROOT, 'outputs')
os.makedirs(out_dir, exist_ok=True)

# extract cells in the problematic segment
cells_idx = np.where(mask_seg)[0]
csv_path = os.path.join(out_dir, 'dry_region_cells.csv')
np.savetxt(csv_path, np.column_stack([coords2[cells_idx,0], coords2[cells_idx,1], depth[cells_idx]]), delimiter=',', header='x,y,depth', comments='')
print('Saved CSV:', csv_path)

# zoomed scatter of depths in that region
fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(coords2[cells_idx,0], coords2[cells_idx,1], c=depth[cells_idx], s=4, cmap='viridis')
ax.set_title('Depths in largest dry segment')
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, label='depth')
fig.savefig(os.path.join(out_dir, 'dry_region_depth_scatter.png'), dpi=150)
plt.close(fig)

# histogram
fig, ax = plt.subplots(figsize=(6,3))
ax.hist(depth[cells_idx], bins=50)
ax.set_title('Depth histogram for problematic dry stretch')
fig.savefig(os.path.join(out_dir, 'dry_region_depth_hist.png'), dpi=150)
plt.close(fig)

print('Saved depth scatter and histogram to outputs/')
