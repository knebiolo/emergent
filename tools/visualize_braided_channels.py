"""
Visualize braided channel structure and compare upstream progress measurement methods.

Shows:
1. Current single longitudinal profile
2. Velocity magnitude (identifies main channels)
3. Flow accumulation (channel network detection)
4. Proposed flow-distance field for upstream progress
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import viridis, plasma, coolwarm
from pathlib import Path
from scipy import ndimage

def compute_flow_accumulation(vel_x, vel_y, threshold=0.1):
    """
    Compute flow accumulation from velocity field.
    Identifies main channels where flow converges.
    """
    # Flow magnitude (handle NaN)
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    
    # Only accumulate where velocity > threshold and not NaN
    active = (vel_mag > threshold) & ~np.isnan(vel_mag)
    
    # Flow direction (where water goes TO, not FROM)
    flow_dir_x = np.zeros_like(vel_x)
    flow_dir_y = np.zeros_like(vel_y)
    flow_dir_x[active] = vel_x[active] / vel_mag[active]
    flow_dir_y[active] = vel_y[active] / vel_mag[active]
    
    # Simple accumulation: sum of upstream contributing area
    # (This is simplified - full implementation would trace flow paths)
    accumulation = ndimage.uniform_filter(vel_mag, size=5)
    accumulation[~active] = 0
    
    return accumulation, vel_mag


def compute_upstream_distance_field(vel_x, vel_y, downstream_boundary_row=-1, cell_size=1.0):
    """
    Compute cumulative distance to downstream boundary following flow paths.
    This is what we'd use for measuring upstream progress.
    
    Uses a simple sweeping algorithm from downstream to upstream.
    """
    rows, cols = vel_x.shape
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    
    # Initialize distance field
    distance = np.full_like(vel_x, np.nan)
    
    # Downstream boundary (e.g., bottom row)
    if downstream_boundary_row == -1:
        downstream_boundary_row = rows - 1
    
    distance[downstream_boundary_row, :] = 0.0
    
    # Sweep upstream (from bottom to top if flow is generally upward/northward)
    # This is simplified - a proper implementation would use Dijkstra or fast marching
    for row in range(downstream_boundary_row - 1, -1, -1):
        for col in range(cols):
            if np.isnan(vel_mag[row, col]) or vel_mag[row, col] < 0.1:  # No flow
                distance[row, col] = np.nan
                continue
            
            # Distance = downstream neighbor + cell_size
            # (Simplified: assumes flow is primarily northward)
            if row < rows - 1:
                distance[row, col] = distance[row + 1, col] + cell_size
            else:
                distance[row, col] = 0.0
    
    return distance


def main():
    # Data directory
    data_dir = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm")
    
    # Load longitudinal profile
    print("Loading longitudinal profile...")
    longitudinal_gdf = gpd.read_file(data_dir / "longitudinal.shp")
    longitudinal_line = longitudinal_gdf.geometry[0]
    
    # Load velocity rasters
    print("Loading velocity rasters...")
    with rasterio.open(data_dir / "vel_x.tif") as src:
        vel_x = src.read(1)
        nodata_x = src.nodata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        
    with rasterio.open(data_dir / "vel_y.tif") as src:
        vel_y = src.read(1)
        nodata_y = src.nodata
    
    with rasterio.open(data_dir / "vel_mag.tif") as src:
        vel_mag_raster = src.read(1)
        nodata_mag = src.nodata
    
    # Mask NoData values
    if nodata_x is not None:
        vel_x = np.where(vel_x == nodata_x, np.nan, vel_x)
    if nodata_y is not None:
        vel_y = np.where(vel_y == nodata_y, np.nan, vel_y)
    if nodata_mag is not None:
        vel_mag_raster = np.where(vel_mag_raster == nodata_mag, np.nan, vel_mag_raster)
    
    print(f"Velocity range: X=[{np.nanmin(vel_x):.2f}, {np.nanmax(vel_x):.2f}], Y=[{np.nanmin(vel_y):.2f}, {np.nanmax(vel_y):.2f}]")
    
    # Load depth for context
    print("Loading depth raster...")
    with rasterio.open(data_dir / "depth.tif") as src:
        depth = src.read(1)
        nodata_depth = src.nodata
        if nodata_depth is not None:
            depth = np.where(depth == nodata_depth, np.nan, depth)
    
    # Compute flow accumulation
    print("Computing flow accumulation...")
    accumulation, vel_mag = compute_flow_accumulation(vel_x, vel_y)
    
    # Compute upstream distance field (simplified)
    print("Computing upstream distance field...")
    cell_size = abs(transform[0])  # Pixel width
    upstream_dist = compute_upstream_distance_field(vel_x, vel_y, cell_size=cell_size)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Braided Channel Analysis: Upstream Progress Measurement Methods', fontsize=16)
    
    # Get extent for plotting
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    # 1. Current longitudinal profile over depth
    ax = axes[0, 0]
    depth_plot = depth.copy()
    depth_plot[depth_plot <= 0] = np.nan
    im1 = ax.imshow(depth_plot, extent=extent, cmap='Blues', alpha=0.6)
    
    # Plot longitudinal line
    x, y = longitudinal_line.xy
    ax.plot(x, y, 'r-', linewidth=3, label='Current Longitudinal Profile')
    ax.set_title('1. Current Method: Single Centerline')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax, label='Depth (m)')
    
    # 2. Velocity magnitude (shows channels)
    ax = axes[0, 1]
    vel_mag_plot = vel_mag.copy()
    vel_mag_plot[vel_mag_plot <= 0] = np.nan
    im2 = ax.imshow(vel_mag_plot, extent=extent, cmap='plasma', vmin=0, vmax=np.nanpercentile(vel_mag_plot, 95))
    ax.plot(x, y, 'r--', linewidth=2, alpha=0.7, label='Single Profile')
    ax.set_title('2. Velocity Magnitude (Reveals Channels)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax, label='Velocity (m/s)')
    
    # 3. Flow accumulation (channel network)
    ax = axes[0, 2]
    accum_plot = accumulation.copy()
    accum_plot[accum_plot <= 0] = np.nan
    im3 = ax.imshow(accum_plot, extent=extent, cmap='viridis', vmin=0, vmax=np.nanpercentile(accum_plot, 98))
    ax.plot(x, y, 'r--', linewidth=2, alpha=0.7)
    ax.set_title('3. Flow Accumulation (Channel Network)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax, label='Accumulated Flow')
    
    # 4. Velocity vectors (flow direction)
    ax = axes[1, 0]
    # Subsample for visualization
    step = max(vel_x.shape[0] // 30, 1)
    vel_mag_bg = vel_mag.copy()
    vel_mag_bg[vel_mag_bg <= 0] = np.nan
    ax.imshow(vel_mag_bg, extent=extent, cmap='Blues', alpha=0.3)
    
    # Create coordinate grids
    rows, cols = vel_x.shape
    y_coords = np.linspace(bounds.top, bounds.bottom, rows)
    x_coords = np.linspace(bounds.left, bounds.right, cols)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Subsample and plot vectors
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    U_sub = vel_x[::step, ::step]
    V_sub = vel_y[::step, ::step]
    
    # Mask zero velocity
    mag_sub = np.sqrt(U_sub**2 + V_sub**2)
    mask = mag_sub > 0.1
    
    ax.quiver(X_sub[mask], Y_sub[mask], U_sub[mask], V_sub[mask], 
             mag_sub[mask], cmap='coolwarm', scale=50, width=0.003, alpha=0.7)
    ax.plot(x, y, 'g-', linewidth=2, label='Longitudinal Profile')
    ax.set_title('4. Flow Vectors (Flow Direction)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Upstream distance field
    ax = axes[1, 1]
    dist_plot = upstream_dist.copy()
    dist_plot[np.isnan(dist_plot)] = np.nan
    im5 = ax.imshow(dist_plot, extent=extent, cmap='magma')
    ax.plot(x, y, 'cyan', linewidth=2, alpha=0.8, label='Single Profile')
    ax.set_title('5. Upstream Distance Field (Proposed Method)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im5, ax=ax, label='Distance from Outlet (m)')
    
    # 6. Example: Fish path and upstream progress calculation
    ax = axes[1, 2]
    ax.imshow(depth_plot, extent=extent, cmap='Blues', alpha=0.4)
    
    # Plot velocity vectors (subsampled)
    step = max(vel_x.shape[0] // 20, 1)
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    U_sub = vel_x[::step, ::step]
    V_sub = vel_y[::step, ::step]
    mag_sub = np.sqrt(U_sub**2 + V_sub**2)
    mask = mag_sub > 0.1
    
    ax.quiver(X_sub[mask], Y_sub[mask], U_sub[mask], V_sub[mask], 
             alpha=0.3, scale=40, width=0.002, color='gray')
    
    # Current longitudinal profile
    ax.plot(x, y, 'r-', linewidth=3, label='Current: Project onto single line')
    
    # Show example fish path that crosses channels
    # (just illustrative)
    ax.set_title('6. Why Single Line Fails in Complex Geometry')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.text(0.5, 0.02, 
           'Single red line = current method (misses lateral movement)\n' +
           'Flow vectors = actual water direction at each point', 
           transform=ax.transAxes, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = data_dir.parent.parent.parent / "figs" / "braided_channel_analysis.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("CHANNEL STRUCTURE ANALYSIS")
    print("="*70)
    print(f"Longitudinal profile length: {longitudinal_line.length:.1f} m")
    print(f"Raster extent: {bounds.right - bounds.left:.1f} m × {bounds.top - bounds.bottom:.1f} m")
    print(f"Cell size: {cell_size:.2f} m")
    print(f"\nVelocity statistics:")
    print(f"  Mean velocity: {np.nanmean(vel_mag):.2f} m/s")
    print(f"  Max velocity: {np.nanmax(vel_mag):.2f} m/s")
    print(f"Velocity statistics:")
    print(f"  Mean velocity: {np.nanmean(vel_mag):.2f} m/s")
    print(f"  Max velocity: {np.nanmax(vel_mag):.2f} m/s")
    print(f"  Cells with flow > 0.1 m/s: {np.sum((vel_mag > 0.1) & ~np.isnan(vel_mag))}")
    
    print(f"\nDepth statistics:")
    print(f"  Mean depth: {np.nanmean(depth):.2f} m")
    print(f"  Max depth: {np.nanmax(depth):.2f} m")
    print(f"  Wetted area: {np.sum(~np.isnan(depth) & (depth > 0)) * cell_size**2:.1f} m²")
    
    
    print("\n" + "="*70)
    print("CURRENT METHOD (SINGLE LONGITUDINAL PROFILE):")
    print("="*70)
    print("distance = longitudinal_profile.project(final_pos) - ")
    print("           longitudinal_profile.project(initial_pos)")
    print("\nPROBLEM: Projects position onto single centerline.")
    print("         Ignores lateral movement, channel switching, and actual")
    print("         water flow direction at fish location.")
    print("="*70)
    
    
    print("\n" + "="*70)
    print("PROPOSED METHOD: FLOW VECTOR INTEGRATION")
    print("="*70)
    print("\nMeasures: 'Distance swum against the current'")
    print("\nAlgorithm (computed each timestep):")
    print("  1. displacement = position[t] - position[t-1]")
    print("  2. velocity_at_fish = sample(vel_x, vel_y) at fish position")
    print("  3. upstream_unit = -velocity_at_fish / |velocity_at_fish|")
    print("  4. progress = displacement · upstream_unit  (dot product)")
    print("  5. cumulative_distance += progress")
    print("\nKey insight:")
    print("  - Positive progress = fish moved against current (upstream)")
    print("  - Negative progress = fish moved with current (downstream/drifting)")
    print("  - Zero progress = fish moved perpendicular to flow")
    print("\nAdvantages:")
    print("  ✓ Works in fast channels, slow pools, and wide shallow sections")
    print("  ✓ Handles braiding, channel switching, and lateral movement")
    print("  ✓ Uses existing vel_x.tif and vel_y.tif (no preprocessing)")
    print("  ✓ Biologically meaningful: effort against current, not Euclidean distance")
    print("\nImplementation:")
    print("  - Add to simulation.py: method to sample velocity at agent positions")
    print("  - Modify rl_training.py: replace shapely.project() with flow integration")
    print("="*70)


if __name__ == "__main__":
    main()
