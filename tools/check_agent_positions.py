"""Check if agents are in valid wetted cells or swimming on land."""
import h5py
import numpy as np
import sys
sys.path.insert(0, 'src')
from emergent.salmon_abm.utils import geo_to_pixel
from affine import Affine

# Load simulation output
f = h5py.File('outputs/boundary_test/nuyakuk_headless_headless.h5', 'r')

# Get depth and distance_to rasters
depth = np.array(f['environment/depth'])
distance_to = np.array(f['environment/distance_to'])
x_coords = np.array(f['environment/x_coords'])
y_coords = np.array(f['environment/y_coords'])

print(f"Depth shape: {depth.shape}")
print(f"Distance_to shape: {distance_to.shape}")
print(f"x_coords range: {x_coords.min():.2f} to {x_coords.max():.2f}")
print(f"y_coords range: {y_coords.min():.2f} to {y_coords.max():.2f}")

# Reconstruct transform from x_coords and y_coords
# x_coords and y_coords are 2D grids - extract pixel size from spacing
if x_coords.shape == depth.shape and y_coords.shape == depth.shape:
    # Pixel width/height
    pixel_width = abs(x_coords[0, 1] - x_coords[0, 0])
    pixel_height = abs(y_coords[1, 0] - y_coords[0, 0])
    # Top-left corner
    x_origin = x_coords[0, 0]
    y_origin = y_coords[0, 0]
    # Construct transform: assumes north-up raster with negative y-pixel size
    if y_coords[1, 0] < y_coords[0, 0]:
        # y decreases down rows (standard north-up)
        transform = Affine(pixel_width, 0, x_origin, 0, -pixel_height, y_origin)
    else:
        transform = Affine(pixel_width, 0, x_origin, 0, pixel_height, y_origin)
    print(f"Reconstructed transform: pixel_size=({pixel_width:.2f}, {pixel_height:.2f}), origin=({x_origin:.2f}, {y_origin:.2f})")
else:
    print("ERROR: x_coords and y_coords don't match depth shape")
    sys.exit(1)

# Get agent positions
print(f"\nAgent data structure:")
print(f"  X shape: {f['X'].shape}")
print(f"  Y shape: {f['Y'].shape}")

# Check how positions are stored
if len(f['X'].shape) == 1:
    # Single agent trajectory over time
    print("  Format: Single agent trajectory (timesteps,)")
    num_timesteps = f['X'].shape[0]
    num_agents = 1
else:
    # Multiple agents
    print("  Format: Multiple agents (timesteps, agents)")
    num_timesteps, num_agents = f['X'].shape

# Sample a few timesteps
timesteps_to_check = [0, 50, 100, 150, 199] if num_timesteps >= 200 else [0, num_timesteps//2, num_timesteps-1]

for t in timesteps_to_check:
    if t >= num_timesteps:
        continue
    
    if num_agents == 1:
        X_agents = np.array([f['X'][t]])
        Y_agents = np.array([f['Y'][t]])
    else:
        X_agents = np.array(f['X'][t])
        Y_agents = np.array(f['Y'][t])
    
    print(f"\nTimestep {t}:")
    print(f"  Num agents: {len(X_agents)}")
    print(f"  X range: {X_agents.min():.2f} to {X_agents.max():.2f}")
    print(f"  Y range: {Y_agents.min():.2f} to {Y_agents.max():.2f}")
    
    # Convert to pixel coordinates
    on_land_count = 0
    near_edge_count = 0
    valid_count = 0
    
    for i, (x, y) in enumerate(zip(X_agents, Y_agents)):
        row, col = geo_to_pixel(x, y, transform)
        
        # Check bounds
        if row < 0 or row >= depth.shape[0] or col < 0 or col >= depth.shape[1]:
            print(f"  Agent {i}: ({x:.1f}, {y:.1f}) -> OUT OF BOUNDS ({row}, {col})")
            on_land_count += 1
            continue
        
        d = depth[row, col]
        dist = distance_to[row, col]
        
        # Check if on land (nodata or very shallow)
        if not np.isfinite(d) or d == -9999 or d <= 0:
            on_land_count += 1
            if i < 3 or on_land_count <= 5:  # Print first few
                print(f"  Agent {i}: ({x:.1f}, {y:.1f}) -> pixel ({row}, {col}), depth={d:.2f} ON LAND!")
        elif dist < 1.0:  # Very close to boundary
            near_edge_count += 1
            if i < 3:
                print(f"  Agent {i}: ({x:.1f}, {y:.1f}) -> pixel ({row}, {col}), depth={d:.2f}m, dist_to_boundary={dist:.2f}m (NEAR EDGE)")
        else:
            valid_count += 1
    
    print(f"  Summary: {valid_count} in water, {near_edge_count} near edge, {on_land_count} ON LAND")
    
    if on_land_count > 0:
        print(f"  ⚠️  {on_land_count}/{len(X_agents)} agents are swimming on land!")

f.close()
