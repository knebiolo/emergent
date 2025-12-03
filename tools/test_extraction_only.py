"""
Minimal test: read distance_to from HDF and run extraction
"""
import sys
sys.path.insert(0, 'src')

import h5py
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.morphology import skeletonize
from skimage.measure import label
from shapely.geometry import LineString
from shapely.ops import linemerge

hdf_path = r"data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf"

# Read distance raster
with h5py.File(hdf_path, 'r') as f:
    if 'environment/distance_to' in f:
        distance = f['environment/distance_to'][:]
        print(f"Loaded distance_to from HDF: shape {distance.shape}")
        print(f"Distance range: {distance.min():.2f} to {distance.max():.2f} m")
        
        # Run extraction - EXACT visualization logic
        footprint_size = 5
        local_max = maximum_filter(distance, size=footprint_size)
        is_ridge = (distance == local_max) & (distance > 0.5)
        skeleton = skeletonize(is_ridge)
        labeled = label(skeleton, connectivity=2)
        
        ridge_pixels = np.sum(is_ridge)
        print(f"Ridge pixels: {ridge_pixels}")
        print(f"Skeleton labels: {labeled.max()}")
        
        # Build LineStrings
        lines = []
        for comp_id in range(1, labeled.max() + 1):
            ys, xs = np.where(labeled == comp_id)
            if len(xs) < 5:
                continue
            # Without transform, just use pixel coords
            coords = list(zip(xs, ys))
            if len(coords) >= 2:
                lines.append(LineString(coords))
        
        print(f"\nComponents >= 5 pixels: {len(lines)}")
        for i, line in enumerate(lines, 1):
            print(f"  Path {i}: {len(line.coords)} pts, {line.length:.1f}px")
        
        if lines:
            merged = linemerge(lines)
            if hasattr(merged, '__iter__'):
                all_lines = list(merged.geoms if hasattr(merged, 'geoms') else merged)
            else:
                all_lines = [merged]
            
            main = max(all_lines, key=lambda l: l.length) if all_lines else None
            print(f"\nTotal centerlines: {len(all_lines)}")
            if main:
                print(f"Main centerline: {main.length:.1f}px")
    else:
        print("No distance_to in HDF - run mapping first")
