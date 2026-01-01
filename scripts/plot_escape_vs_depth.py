#!/usr/bin/env python
"""Plot escape location over depth raster to see if fish are on land."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
import geopandas as gpd

# Load trace data
df = pd.read_csv('outputs/production_5000agents_900sec_lowcollision/nuyakuk_headless_trace.csv')
last_t = df["timestep"].max()
final = df[df["timestep"] == last_t]

# Load depth raster
with rasterio.open('data/salmon_abm/depth.tif') as src:
    depth = src.read(1)
    transform = src.transform
    bounds = src.bounds
    
# Load starting polygon
poly_gdf = gpd.read_file('data/salmon_abm/start_loc_combined.shp')

# Escape location from error
escape_x, escape_y = 549719.05, 6641559.32

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Plot depth raster - nodata as red, shallow as light blue, deep as dark blue
depth_plot = np.where(np.abs(depth) > 9990, np.nan, depth)  # nodata as NaN
img = ax.imshow(depth_plot, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                cmap='Blues_r', alpha=0.7, vmin=0, vmax=5)
plt.colorbar(img, ax=ax, label='Depth (m)')

# Overlay nodata regions in red
nodata_mask = np.abs(depth) > 9990
nodata_plot = np.where(nodata_mask, 1, np.nan)
ax.imshow(nodata_plot, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
          cmap=ListedColormap(['red']), alpha=0.5)

# Plot starting polygon
poly_gdf.boundary.plot(ax=ax, color='green', linewidth=2, label='Start polygon')

# Plot final agent positions
ax.scatter(final["x"], final["y"], s=1, alpha=0.3, color='yellow', label=f'Agents at t={last_t}')

# Mark escape location
ax.scatter([escape_x], [escape_y], s=200, color='red', marker='X', 
           edgecolors='black', linewidths=2, label='Escape location', zorder=10)

# Add text showing escape location
ax.text(escape_x + 10, escape_y + 10, f'Escape: ({escape_x:.1f}, {escape_y:.1f})',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Fish escape location over depth raster (t={last_t})\nRed = Dry land (nodata), Blue = Water')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('outputs/production_5000agents_900sec_lowcollision/escape_vs_depth.png', dpi=150, bbox_inches='tight')
print(f'Saved plot to: outputs/production_5000agents_900sec_lowcollision/escape_vs_depth.png')
plt.show()

# Sample depth at escape location
from affine import Affine
if isinstance(transform, Affine):
    inv_transform = ~transform
else:
    inv_transform = ~Affine.from_gdal(*transform)

col, row = inv_transform * (escape_x, escape_y)
row, col = int(row), int(col)
H, W = depth.shape

if 0 <= row < H and 0 <= col < W:
    escape_depth = depth[row, col]
    print(f'\nDepth at escape location: {escape_depth:.2f} m')
    if abs(escape_depth) > 9990:
        print('  -> NODATA - Fish is on DRY LAND!')
    else:
        print(f'  -> Fish is in {escape_depth:.2f}m of water (valid depth)')
else:
    print(f'\nEscape location is OUT OF BOUNDS of depth raster!')
