"""Multi-path ridge extraction: find main + side channels via iterative Dijkstra."""
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import maximum_filter
from skimage.morphology import remove_small_objects
from shapely.geometry import LineString
import geopandas as gpd
from pathlib import Path
from heapq import heappush, heappop

raster_dir = Path(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\outputs\hecras_run')
distance_path = raster_dir / 'distance_to.tif'

with rasterio.open(distance_path) as src:
    distance_to = src.read(1)
    transform = src.transform
    crs = src.crs

print(f'Raster shape: {distance_to.shape}')
distance_to = np.nan_to_num(distance_to, nan=0.0, posinf=0.0, neginf=0.0)
distance_to = np.clip(distance_to, 0.0, 100.0)
print(f'Distance range: {distance_to.min():.2f} to {distance_to.max():.2f} m')

# Find ridge: local maxima of distance_to
footprint = 9
local_max = maximum_filter(distance_to, size=footprint)
is_ridge = (distance_to == local_max) & (distance_to > 1.5)
is_ridge = remove_small_objects(is_ridge, min_size=100)
print(f'Ridge pixels: {is_ridge.sum()}')

cost = 1.0 / (distance_to + 0.1)

def dijkstra_path(start, end, is_ridge_available, cost):
    """Find least-cost path, staying on available ridge where possible."""
    rows, cols = cost.shape
    visited = np.zeros_like(cost, dtype=bool)
    came_from = {}
    g_score = np.full_like(cost, np.inf)
    g_score[start] = 0.0
    
    heap = [(0.0, start)]
    ndirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    while heap:
        current_cost, current = heappop(heap)
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
        
        if visited[current]:
            continue
        visited[current] = True
        
        r, c = current
        for dr, dc in ndirs:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
                continue
            neighbor = (rr, cc)
            if visited[neighbor]:
                continue
            
            edge_cost = cost[rr, cc]
            if not is_ridge_available[rr, cc]:
                edge_cost *= 10.0
            
            tentative_g = g_score[current] + edge_cost
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heappush(heap, (tentative_g, neighbor))
    
    return None

# Multi-path extraction
centerlines = []
ridge_available = is_ridge.copy()
max_paths = 10
min_path_length = 200  # meters

for iteration in range(max_paths):
    ridge_coords = np.argwhere(ridge_available)
    if len(ridge_coords) < 100:
        break
    
    # Find furthest endpoints in available ridge
    row_min_idx = np.argmin(ridge_coords[:, 0])
    row_max_idx = np.argmax(ridge_coords[:, 0])
    start = tuple(ridge_coords[row_min_idx])
    end = tuple(ridge_coords[row_max_idx])
    
    path = dijkstra_path(start, end, ridge_available, cost)
    
    if path is None or len(path) < 10:
        break
    
    world_coords = [(transform * (c, r)) for (r, c) in path]
    line = LineString(world_coords)
    
    if line.length < min_path_length:
        break
    
    centerlines.append(line)
    print(f'  Path {iteration+1}: {len(path)} pts, {line.length:.1f}m')
    
    # Remove this path from available ridge (with small buffer)
    for (r, c) in path:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < ridge_available.shape[0] and 0 <= cc < ridge_available.shape[1]:
                    ridge_available[rr, cc] = False

print(f'\nTotal centerlines: {len(centerlines)}')
if centerlines:
    main = max(centerlines, key=lambda g: g.length)
    print(f'Main centerline: {main.length:.1f}m')

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0,0]
im = ax.imshow(distance_to, cmap='viridis', interpolation='nearest')
ax.set_title('Distance to Edge', fontsize=14, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[0,1]
ax.imshow(distance_to, cmap='gray', alpha=0.3)
ax.imshow(is_ridge, cmap='Reds', alpha=0.7)
ax.set_title(f'Ridge ({is_ridge.sum()} px)', fontsize=14, fontweight='bold')
ax.axis('off')

ax = axes[1,0]
ax.imshow(distance_to, cmap='Blues', alpha=0.5)
for i, line in enumerate(centerlines):
    xs, ys = line.xy
    pixel_coords = [~transform * (x, y) for x, y in zip(xs, ys)]
    px, py = [c[0] for c in pixel_coords], [c[1] for c in pixel_coords]
    ax.plot(px, py, linewidth=2, alpha=0.8, label=f'{i+1}: {line.length:.0f}m')
ax.set_title(f'All Centerlines (n={len(centerlines)})', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.axis('off')

ax = axes[1,1]
ax.imshow(distance_to, cmap='Blues', alpha=0.5)
if centerlines:
    main = max(centerlines, key=lambda g: g.length)
    xs, ys = main.xy
    pixel_coords = [~transform * (x, y) for x, y in zip(xs, ys)]
    px, py = [c[0] for c in pixel_coords], [c[1] for c in pixel_coords]
    ax.plot(px, py, 'r-', linewidth=3, alpha=0.9)
    ax.set_title(f'Main ({main.length:.0f}m)', fontsize=14, fontweight='bold')
else:
    ax.set_title('No Centerline', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
out_path = Path(r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\figs\centerline_extraction.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')

if centerlines:
    gdf = gpd.GeoDataFrame([{'id': i, 'length_m': line.length, 'geometry': line} for i, line in enumerate(centerlines)], crs=crs)
    shp_path = raster_dir / 'extracted_centerlines.shp'
    gdf.to_file(shp_path)
    print(f'Saved: {shp_path}')

plt.show()
