"""Visualize centerline extraction from distance_to raster."""
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import maximum_filter
from skimage.morphology import skeletonize
from skimage.measure import label
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import geopandas as gpd
from pathlib import Path

# Load the distance_to raster
raster_path = Path(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\hecras_rasters\distance_to.tif')

with rasterio.open(raster_path) as src:
    distance_to_rast = src.read(1)
    transform = src.transform
    crs = src.crs
    
print(f'Raster shape: {distance_to_rast.shape}')
print(f'Distance range: {distance_to_rast.min():.2f} to {distance_to_rast.max():.2f} meters')

# Find local maxima (ridge detection)
footprint_size = 5
local_max = maximum_filter(distance_to_rast, size=footprint_size)
is_ridge = (distance_to_rast == local_max) & (distance_to_rast > 0.5)

print(f'Ridge pixels: {is_ridge.sum()}')

# Skeletonize
skeleton = skeletonize(is_ridge)
print(f'Skeleton pixels: {skeleton.sum()}')

# Convert skeleton to LineString(s)
labeled = label(skeleton, connectivity=2)
print(f'Number of connected components: {labeled.max()}')

centerlines = []
for region_id in range(1, labeled.max() + 1):
    region_mask = (labeled == region_id)
    ys, xs = np.where(region_mask)
    
    if len(xs) < 5:  # Skip very short segments
        continue
    
    # Convert pixel coords to world coords
    world_coords = []
    for i in range(len(xs)):
        x_world, y_world = transform * (xs[i], ys[i])
        world_coords.append((x_world, y_world))
    
    if len(world_coords) >= 2:
        line = LineString(world_coords)
        centerlines.append(line)
        print(f'  Component {region_id}: {len(world_coords)} points, {line.length:.1f}m long')

# Merge and find main centerline
if centerlines:
    merged = linemerge(centerlines)
    if isinstance(merged, LineString):
        main_centerline = merged
        all_lines = [merged]
    elif isinstance(merged, MultiLineString):
        main_centerline = max(merged.geoms, key=lambda g: g.length)
        all_lines = list(merged.geoms)
    else:
        main_centerline = None
        all_lines = []
    
    print(f'\nTotal centerlines after merge: {len(all_lines)}')
    if main_centerline:
        print(f'Main centerline length: {main_centerline.length:.1f}m')

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Original distance_to raster
ax = axes[0, 0]
im1 = ax.imshow(distance_to_rast, cmap='viridis', interpolation='nearest')
ax.set_title('Distance to Edge (meters)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

# 2. Ridge detection (local maxima)
ax = axes[0, 1]
ax.imshow(distance_to_rast, cmap='gray', alpha=0.3)
ax.imshow(is_ridge, cmap='Reds', alpha=0.7)
ax.set_title('Ridge Detection (Local Maxima)', fontsize=14, fontweight='bold')
ax.axis('off')

# 3. Skeletonized centerlines
ax = axes[0, 2]
ax.imshow(distance_to_rast, cmap='gray', alpha=0.3)
ax.imshow(skeleton, cmap='hot', alpha=0.8)
ax.set_title('Skeletonized Centerlines', fontsize=14, fontweight='bold')
ax.axis('off')

# 4. Labeled components
ax = axes[1, 0]
ax.imshow(distance_to_rast, cmap='gray', alpha=0.3)
im4 = ax.imshow(labeled, cmap='tab20', alpha=0.7)
ax.set_title(f'Connected Components (n={labeled.max()})', fontsize=14, fontweight='bold')
ax.axis('off')

# 5. All extracted centerlines
ax = axes[1, 1]
ax.imshow(distance_to_rast, cmap='Blues', alpha=0.5)
if centerlines:
    for i, line in enumerate(all_lines):
        xs, ys = line.xy
        # Convert world coords back to pixel coords for plotting
        pixel_coords = [~transform * (x, y) for x, y in zip(xs, ys)]
        px = [c[0] for c in pixel_coords]
        py = [c[1] for c in pixel_coords]
        ax.plot(px, py, 'r-', linewidth=2, alpha=0.7)
ax.set_title(f'All Centerlines (n={len(all_lines)})', fontsize=14, fontweight='bold')
ax.axis('off')

# 6. Main centerline only
ax = axes[1, 2]
ax.imshow(distance_to_rast, cmap='Blues', alpha=0.5)
if main_centerline:
    xs, ys = main_centerline.xy
    pixel_coords = [~transform * (x, y) for x, y in zip(xs, ys)]
    px = [c[0] for c in pixel_coords]
    py = [c[1] for c in pixel_coords]
    ax.plot(px, py, 'r-', linewidth=3, alpha=0.9)
    ax.set_title(f'Main Centerline ({main_centerline.length:.0f}m)', fontsize=14, fontweight='bold')
else:
    ax.set_title('No Main Centerline', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
out_path = Path(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\figs\centerline_extraction.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nVisualization saved to: {out_path}')

# Save centerlines as shapefile for inspection
if centerlines:
    gdf = gpd.GeoDataFrame(
        [{'id': i, 'length_m': line.length, 'geometry': line} 
         for i, line in enumerate(all_lines)],
        crs=crs
    )
    shp_path = Path(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\hecras_rasters\extracted_centerlines.shp')
    gdf.to_file(shp_path)
    print(f'Centerlines shapefile saved to: {shp_path}')

plt.show()
