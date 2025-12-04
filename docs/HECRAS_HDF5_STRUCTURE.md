# HEC-RAS HDF5 File Structure and Navigation Guide

**Document Purpose**: Reference guide for understanding HEC-RAS plan HDF5 files and how the emergent salmon ABM uses them.

**Example File**: `C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf`

**Last Updated**: December 3, 2025

---

## Table of Contents
1. [File Overview](#file-overview)
2. [HEC-RAS Native Structure](#hec-ras-native-structure)
3. [ABM-Created Datasets](#abm-created-datasets)
4. [Wetted Perimeter Inference](#wetted-perimeter-inference)
5. [Key Initialization Steps](#key-initialization-steps)
6. [Code Navigation](#code-navigation)
7. [Performance Considerations](#performance-considerations)

---

## File Overview

### Quick Stats (Nuyakuk Example)
- **Total HECRAS Cells**: 942,280 (irregular mesh)
- **Timesteps**: 121
- **Spatial Extent**: ~2.6 km x 1.2 km
- **ABM Raster Grid**: 2318 x 5175 cells at 0.5m resolution
- **Wetted Cells** (depth > 0.05m at t=0): 223,312 (23.7%)
- **Dry/Shallow Cells**: 718,968 (76.3%)

### File Structure Tree
```
HECRAS_Plan.hdf
├── Event Conditions/
├── Geometry/
│   ├── 2D Flow Areas/
│   │   └── 2D area/
│   │       ├── Cells Center Coordinate        [N x 2] - Cell centers (X, Y)
│   │       ├── Cells Minimum Elevation        [N] - Cell bed elevation
│   │       ├── Cells Face and Orientation Info [N x 2] - Connectivity
│   │       ├── FacePoints Coordinate          [M x 2] - Face vertices
│   │       ├── FacePoints Is Perimeter        [M] - Boundary flag (-1=perimeter)
│   │       └── Perimeter                      [P x 2] - Domain perimeter points
│   └── ...
├── Results/
│   └── Unsteady/
│       └── Output/
│           └── Output Blocks/
│               └── Base Output/
│                   └── Unsteady Time Series/
│                       ├── Time                        [T] - Time values
│                       └── 2D Flow Areas/
│                           └── 2D area/
│                               ├── Cell Hydraulic Depth       [T x N]
│                               ├── Cell Velocity - Velocity X [T x N]
│                               ├── Cell Velocity - Velocity Y [T x N]
│                               ├── Water Surface              [T x N]
│                               └── ...
├── x_coords                    [H x W] - ABM regular grid X (created by ABM)
├── y_coords                    [H x W] - ABM regular grid Y (created by ABM)
└── environment/                         - ABM rasterized fields (created by ABM)
    ├── depth                   [H x W]
    ├── vel_x                   [H x W]
    ├── vel_y                   [H x W]
    ├── vel_mag                 [H x W]
    └── vel_dir                 [H x W]
```

Where:
- `N` = number of HECRAS cells (e.g., 942,280)
- `M` = number of face points (e.g., 950,530)
- `T` = number of timesteps (e.g., 121)
- `H x W` = ABM grid dimensions (e.g., 2318 x 5175)
- `P` = perimeter points (e.g., 125)

---

## HEC-RAS Native Structure

### Geometry Group

#### 1. Cell Centers (`Geometry/2D Flow Areas/2D area/Cells Center Coordinate`)
- **Shape**: `(N, 2)` where N = total cells
- **Dtype**: `float64`
- **Content**: `[X, Y]` coordinates of irregular cell centers
- **Usage**: Primary geometry for KDTree mapping
- **Example**:
  ```python
  coords = hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]
  # coords.shape = (942280, 2)
  # coords[0] = [549363.30322771, 6641868.87398129]
  ```

#### 2. Cell Elevation (`Geometry/2D Flow Areas/2D area/Cells Minimum Elevation`)
- **Shape**: `(N,)`
- **Dtype**: `float64`
- **Content**: Bed elevation at each cell (may contain NaN for inactive cells)
- **Usage**: Bathymetry, not directly used in ABM (uses depth from results)
- **Note**: ~99.6% of cells have valid elevations

#### 3. Cell Connectivity (`Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info`)
- **Shape**: `(N, 2)`
- **Dtype**: `int32`
- **Content**: `[start_index, num_faces]` for each cell
- **Usage**: Maps cells to their face points for mesh reconstruction
- **Example**:
  ```python
  face_info = hdf['Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info'][:]
  # face_info[0] = [0, 5]  -> Cell 0 has 5 faces starting at index 0
  # face_info[1] = [5, 4]  -> Cell 1 has 4 faces starting at index 5
  ```

#### 4. Face Points (`Geometry/2D Flow Areas/2D area/FacePoints Coordinate`)
- **Shape**: `(M, 2)` where M > N (vertices shared between cells)
- **Dtype**: `float64`
- **Content**: `[X, Y]` coordinates of cell face vertices
- **Usage**: Define cell polygon boundaries (needed for distance-to-bank)

#### 5. Perimeter Flag (`Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter`)
- **Shape**: `(M,)`
- **Dtype**: `int32`
- **Content**: `-1` = perimeter point, `0` = interior point
- **Usage**: Identify boundary edges for wetted perimeter calculation
- **Stats**: 3,833 perimeter points out of 950,530 total

#### 6. Domain Perimeter (`Geometry/2D Flow Areas/2D area/Perimeter`)
- **Shape**: `(P, 2)`
- **Dtype**: `float64`
- **Content**: Ordered perimeter coordinates defining domain boundary
- **Usage**: Domain extent, not necessarily wetted perimeter

### Results Group

#### Time Array (`Results/Unsteady/.../Time`)
- **Shape**: `(T,)`
- **Dtype**: `float64`
- **Content**: Time values in days (fractional)
- **Example**: `[0.0, 0.00069444, 0.00138889, ...]` (121 steps)

#### Time-Varying Fields (`Results/Unsteady/.../2D Flow Areas/2D area/`)

All time-varying fields have shape `(T, N)`:

1. **Cell Hydraulic Depth** (`Cell Hydraulic Depth`)
   - Water depth above bed elevation
   - Units: meters
   - **Critical**: Used to define wetted perimeter at t=0

2. **Velocity X** (`Cell Velocity - Velocity X`)
   - Easting velocity component
   - Units: m/s
   - Range example: -1.18 to 3.95 m/s

3. **Velocity Y** (`Cell Velocity - Velocity Y`)
   - Northing velocity component
   - Units: m/s
   - Range example: -2.23 to 2.02 m/s

4. **Water Surface** (`Water Surface`)
   - Water surface elevation
   - Units: meters (absolute elevation)
   - Depth = Water Surface - Cells Minimum Elevation

5. **Cell Invert Depth** (`Cell Invert Depth`)
   - Depth relative to cell invert
   - Usually same as Hydraulic Depth

---

## ABM-Created Datasets

These datasets are **NOT** part of the original HECRAS file. The ABM creates them during initialization to provide a regular raster grid interface.

### 1. `x_coords` and `y_coords`
- **Shape**: `(H, W)` - 2D arrays
- **Dtype**: `float32`
- **Content**: Regular grid coordinates
- **Creation**: `ensure_hdf_coords_from_hecras()` function
- **Purpose**: Enable raster-based sampling for legacy code
- **Example**:
  ```python
  x_coords = hdf['x_coords'][:]  # Shape: (2318, 5175)
  y_coords = hdf['y_coords'][:]  # Shape: (2318, 5175)
  # Each [i, j] gives the center coordinate of that grid cell
  ```

### 2. `environment/` Group
- **Datasets**: `depth`, `vel_x`, `vel_y`, `vel_mag`, `vel_dir`
- **Shape**: All `(H, W)`
- **Dtype**: `float32`
- **Creation**: `map_hecras_to_env_rasters()` function (called each timestep)
- **Purpose**: Rasterized HECRAS fields via IDW interpolation
- **Update Frequency**: Every timestep (time-varying)
- **Method**: KDTree nearest-neighbor IDW from HECRAS cells to regular grid

---

## Wetted Perimeter Inference

**Problem**: HECRAS does not provide a "wetted perimeter" dataset. Must be inferred.

### Algorithm

1. **Sample depth at t=0**:
   ```python
   depth_t0 = hdf['Results/.../Cell Hydraulic Depth'][0]  # First timestep
   ```

2. **Apply wetted threshold**:
   ```python
   wetted = depth_t0 > 0.05  # meters (configurable threshold)
   dry = depth_t0 <= 0.05
   ```

3. **Remove islands** (dry regions completely surrounded by wetted cells):
   - Islands are dry cells with no path to domain perimeter
   - Use connected components on dry cells
   - Any dry region touching the domain boundary is NOT an island
   - Algorithm:
     ```python
     from scipy.ndimage import label
     
     # Label connected dry regions
     dry_labeled, num_features = label(dry_mask)
     
     # Identify which regions touch the boundary
     boundary_mask = get_boundary_mask(coords, perimeter)
     touching_boundary = np.unique(dry_labeled[boundary_mask])
     
     # Islands are dry regions NOT touching boundary
     for region_id in range(1, num_features + 1):
         if region_id not in touching_boundary:
             # This is an island - mark as wetted
             wetted[dry_labeled == region_id] = True
     ```

4. **Extract wetted perimeter**:
   - Wetted perimeter = boundary edges between wetted and dry cells
   - Use `FacePoints Is Perimeter` to filter to domain edges
   - Only keep perimeter points where adjacent cell is wetted

### Implementation Notes

- **Threshold**: 0.05m is standard but should be configurable
- **Island removal**: Critical for realistic channel geometry
- **Boundary definition**: Use `Geometry/.../FacePoints Is Perimeter == -1`
- **Performance**: Island removal requires graph traversal (slower for large meshes)

### Stats for Nuyakuk Example
- Total cells: 942,280
- Wetted (>0.05m): 223,312 (23.7%)
- Dry/shallow (≤0.05m): 718,968 (76.3%)
- Completely dry (=0): 685,904 (72.8%)

---

## Key Initialization Steps

### HECRAS Mode Initialization Flow

```
1. Load HECRAS Plan HDF5
   ├─ Read cell centers → Build KDTree (fast nearest-neighbor)
   ├─ Read cell elevations (optional, for reference)
   └─ Cache KDTree on simulation object

2. Infer Wetted Perimeter (t=0)
   ├─ Sample depth at first timestep
   ├─ Apply threshold (0.05m)
   ├─ Remove islands
   └─ Extract boundary between wetted/dry

3. Compute Distance-to-Bank
   ├─ Input: Wetted perimeter edges
   ├─ Method: Dijkstra or distance transform on wetted cells
   └─ Output: Distance raster (used for centerline)

4. Derive Centerline
   ├─ Input: Distance-to-bank raster
   ├─ Method: Ridge detection + skeletonization
   └─ Output: LineString (used for upstream progress)

5. Create ABM Raster Grid (optional, for legacy compatibility)
   ├─ Compute affine transform from HECRAS cell spacing
   ├─ Create x_coords, y_coords datasets
   └─ Write to HDF5 (enables raster-based code paths)

6. Initialize Agents
   ├─ Sample starting positions from polygon
   ├─ Precompute KDTree mapping for initial positions
   └─ Ready for simulation

7. Timestep Loop
   ├─ Update HECRAS mapping (if agents moved significantly)
   ├─ Map time-varying fields (depth, velocity) via KDTree
   ├─ Optionally rasterize to environment/ group
   └─ Agent behaviors use mapped values
```

### Functions Involved

| Step | Function | File Location |
|------|----------|---------------|
| Load HECRAS | `load_hecras_plan_cached()` | Line 748 |
| Build KDTree | `HECRASMap.__init__()` | Line 634 |
| Map fields | `HECRASMap.map_idw()` | Line 741 |
| Infer wetted | **NEEDS IMPLEMENTATION** | - |
| Distance-to-bank | `compute_alongstream_raster()` | Line 2069 |
| Centerline | `derive_centerline_from_distance_raster()` | Line 1135 |
| Rasterize | `map_hecras_to_env_rasters()` | Line 860 |
| Create coords | `ensure_hdf_coords_from_hecras()` | Line 788 |

---

## Code Navigation

### Reading HECRAS Geometry

```python
import h5py
hdf = h5py.File('plan.hdf', 'r')

# Cell centers (main geometry)
coords = hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:]

# Cell elevations
elev = hdf['Geometry/2D Flow Areas/2D area/Cells Minimum Elevation'][:]

# Perimeter
perim = hdf['Geometry/2D Flow Areas/2D area/Perimeter'][:]

# Face points (for cell boundaries)
facepoints = hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'][:]
is_perim = hdf['Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter'][:]
```

### Reading Time-Varying Results

```python
# Get timestep 0
depth = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0]

# Get all timesteps for velocity
vel_x = hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Velocity - Velocity X'][:]
```

### Using HECRASMap Class

```python
from sockeye_SoA_OpenGL_RL import HECRASMap

# Load plan and build KDTree
m = HECRASMap(
    plan_path='plan.hdf',
    field_names=['Cell Hydraulic Depth', 'Cell Velocity - Velocity X']
)

# Map to agent positions
agent_xy = np.column_stack((sim.X, sim.Y))
mapped_values = m.map_idw(agent_xy, k=8)  # Returns dict with fields

# Access specific field
depth_at_agents = mapped_values['Cell Hydraulic Depth']
```

### Efficient Batch Sampling

```python
# Map multiple fields at once
fields = ['Cell Hydraulic Depth', 
          'Cell Velocity - Velocity X',
          'Cell Velocity - Velocity Y']

m = HECRASMap(plan_path, field_names=fields)
values = m.map_idw(agent_positions, k=8)

depth = values['Cell Hydraulic Depth']
vel_x = values['Cell Velocity - Velocity X']
vel_y = values['Cell Velocity - Velocity Y']
```

---

## Performance Considerations

### KDTree Efficiency

- **Build time**: O(N log N) - cached per simulation
- **Query time**: O(k log N) per agent per field
- **Memory**: ~150 MB for 1M cells (coords + tree)

### Recommended k Values

| Use Case | k | Rationale |
|----------|---|-----------|
| Time-varying sampling | 1-3 | Speed critical, nearest is often sufficient |
| Raster interpolation | 8-12 | Smoother fields, computed once per timestep |
| Distance-to-bank | 1 | Binary classification, nearest is exact |

### Caching Strategy

```python
# Cache KDTree on simulation object (done automatically)
if not hasattr(simulation, '_hecras_maps'):
    simulation._hecras_maps = {}

# Cache inverse affine transforms
if not hasattr(simulation, '_inv_transform_cache'):
    simulation._inv_transform_cache = {}
```

### Memory Management

- **Avoid**: Loading all timesteps into memory
- **Do**: Use HDF5 slicing `[timestep_idx]` to load one timestep at a time
- **Cache**: KDTree, affine transforms, static geometry
- **Don't cache**: Time-varying fields (depth, velocity)

---

## Current Issues and Fixes Needed

### 1. ✗ Wetted Perimeter Inference Not Implemented

**Problem**: No function to infer wetted perimeter from depth threshold

**Fix Needed**: Implement `infer_wetted_perimeter_from_hecras()`
```python
def infer_wetted_perimeter_from_hecras(hdf, threshold=0.05):
    """
    Infer wetted perimeter from HECRAS depth at t=0.
    
    Returns:
        wetted_mask: (N,) boolean array
        perimeter_edges: List of (x, y) coordinates defining wetted boundary
    """
    pass
```

### 2. ✗ Distance-to-Bank Logic Incompatible

**Problem**: `compute_alongstream_raster()` expects regular raster, but HECRAS is irregular

**Fix Needed**: 
- Option A: Compute distance-to-bank on HECRAS cells directly using mesh edges
- Option B: Rasterize wetted mask first, then compute distance on raster

### 3. ✗ Centerline Derivation Missing Input

**Problem**: `derive_centerline_from_distance_raster()` needs distance raster (doesn't exist yet)

**Fix Needed**: Call after distance-to-bank is computed

### 4. ✗ Initialization Order Wrong

**Problem**: ABM tries to create `x_coords`/`y_coords` before determining wetted area

**Fix Needed**: Reorder initialization:
```python
# Current (wrong):
1. create x_coords/y_coords
2. map HECRAS to rasters
3. ??? wetted perimeter never computed

# Correct:
1. Load HECRAS geometry + build KDTree
2. Infer wetted perimeter from depth(t=0)
3. Compute distance-to-bank on HECRAS mesh
4. Derive centerline from distance field
5. Create x_coords/y_coords (optional)
6. Initialize agents
```

---

## Quick Reference Commands

### Inspect HDF5 Structure
```bash
python -c "import h5py; hdf = h5py.File('plan.hdf', 'r'); print(list(hdf.keys()))"
```

### Check File Size
```bash
ls -lh plan.hdf
```

### Verify HECRAS Fields Exist
```python
required_fields = [
    'Geometry/2D Flow Areas/2D area/Cells Center Coordinate',
    'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
]
for field in required_fields:
    assert field in hdf, f"Missing: {field}"
```

---

## Summary Checklist for HECRAS Mode

- [x] Load HECRAS plan HDF5
- [x] Build KDTree from cell centers
- [x] Cache KDTree on simulation
- [ ] **Infer wetted perimeter** (MISSING)
- [ ] **Compute distance-to-bank on irregular mesh** (NEEDS FIX)
- [ ] **Derive centerline from distance field** (DEPENDS ON ABOVE)
- [x] Map time-varying fields via KDTree
- [x] Optionally rasterize to environment/ group
- [x] Sample environment for agents

**Priority**: Implement wetted perimeter inference first, then distance-to-bank on irregular mesh.

---

## Appendix: Field Name Mappings

ABM uses normalized short names. Mapping to HECRAS field names:

| ABM Name | HECRAS Field Name |
|----------|-------------------|
| `depth` | `Cell Hydraulic Depth` |
| `vel_x` | `Cell Velocity - Velocity X` |
| `vel_y` | `Cell Velocity - Velocity Y` |
| `wsel` | `Water Surface` |
| `elev` | `Cells Minimum Elevation` (geometry, not results) |

Computed fields:
- `vel_mag` = sqrt(vel_x² + vel_y²)
- `vel_dir` = atan2(vel_y, vel_x)

---

**End of Document**
