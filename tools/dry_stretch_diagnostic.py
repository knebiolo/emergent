"""dry_stretch_diagnostic.py
Windows-friendly script to analyze and visualize long dry stretches.
Saves outputs/dry_stretch_profile.png and outputs/dry_stretch_map.png
"""
import os, sys, numpy as np
import matplotlib.pyplot as plt
import h5py

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# locate hdf
data_dir = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
if not os.path.exists(data_dir):
    print('HECRAS data dir not found:', data_dir)
    sys.exit(1)

hdf_path = None
for f in os.listdir(data_dir):
    if f.endswith('.p05.hdf'):
        hdf_path = os.path.join(data_dir, f)
        break
if hdf_path is None:
    print('HECRAS HDF not found in', data_dir)
    sys.exit(1)

with h5py.File(hdf_path, 'r') as hdf:
    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])

coords2 = coords[:, :2]
depth_thresh = 0.05
wetted = depth > depth_thresh

# PCA to find main axis
mean = coords2.mean(axis=0)
X = coords2 - mean
u, s, vt = np.linalg.svd(X, full_matrices=False)
pc1 = vt[0]
proj = X.dot(pc1)

# create bins along axis
n_bins = 300
bins = np.linspace(proj.min(), proj.max(), n_bins+1)
bin_idx = np.digitize(proj, bins) - 1
frac_wetted = np.zeros(n_bins)
counts = np.zeros(n_bins, dtype=int)
for i in range(n_bins):
    mask = bin_idx == i
    counts[i] = np.sum(mask)
    if counts[i] == 0:
        frac_wetted[i] = np.nan
    else:
        frac_wetted[i] = np.sum(wetted[mask]) / counts[i]

# find long contiguous run where frac_wetted < threshold (e.g., <0.2)
thr = 0.2
is_dry_bin = np.where(np.nan_to_num(frac_wetted, nan=0.0) < thr, 1, 0)
# find contiguous segments of is_dry_bin==1
segments = []
start = None
for i, v in enumerate(is_dry_bin):
    if v == 1 and start is None:
        start = i
    if v == 0 and start is not None:
        segments.append((start, i-1))
        start = None
if start is not None:
    segments.append((start, n_bins-1))

segments_sorted = sorted(segments, key=lambda s: s[1]-s[0], reverse=True)

out_dir = os.path.join(REPO_ROOT, 'outputs')
os.makedirs(out_dir, exist_ok=True)

# save profile plot
fig, ax = plt.subplots(figsize=(10,4))
bin_centers = 0.5*(bins[:-1]+bins[1:])
ax.plot(bin_centers, frac_wetted, '-k')
ax.axhline(thr, color='r', linestyle='--', label=f'dry threshold={thr}')
ax.set_xlabel('Distance along main axis (proj units)')
ax.set_ylabel('Wetted fraction (bin)')
ax.set_title('Wetted fraction along main axis')
for (s,e) in segments_sorted[:3]:
    ax.axvspan(bins[s], bins[e+1], color='orange', alpha=0.2)
ax.legend()
fig.savefig(os.path.join(out_dir, 'dry_stretch_profile.png'), dpi=150)
plt.close(fig)

# map plot: show wetted/dry and highlight largest dry segment bins
largest_seg = segments_sorted[0] if len(segments_sorted)>0 else None
highlight_mask = np.zeros(len(coords2), dtype=bool)
if largest_seg is not None:
    s,e = largest_seg
    bins_mask = (bin_idx >= s) & (bin_idx <= e)
    highlight_mask = bins_mask & (~wetted)

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(coords2[~wetted,0], coords2[~wetted,1], s=0.5, c='0.6', label='dry')
ax.scatter(coords2[wetted,0], coords2[wetted,1], s=0.5, c='blue', label='wetted')
if largest_seg is not None and np.any(highlight_mask):
    ax.scatter(coords2[highlight_mask,0], coords2[highlight_mask,1], s=2, c='red', label='problem dry stretch')
ax.set_aspect('equal')
ax.set_title('Wetted (blue) vs dry (gray); problematic dry stretch in red')
ax.legend(markerscale=4)
fig.savefig(os.path.join(out_dir, 'dry_stretch_map.png'), dpi=150)
plt.close(fig)

# print summary
print('Total wetted cells:', np.sum(wetted))
print('Total dry cells:', np.sum(~wetted))
if largest_seg is not None:
    s,e = largest_seg
    print('Largest dry segment bins:', s, e, 'bin count approx=', e-s+1)
    length = bins[e+1] - bins[s]
    print('Approx segment length (proj units):', length)
else:
    print('No dry segments found under threshold')
print('Saved outputs:', os.path.join(out_dir, 'dry_stretch_profile.png'), os.path.join(out_dir, 'dry_stretch_map.png'))
