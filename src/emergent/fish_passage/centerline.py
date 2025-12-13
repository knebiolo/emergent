import numpy as np
from shapely.geometry import LineString
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


def derive_centerline_from_hecras_distance(coords, distances, wetted_mask, crs=None, min_distance_threshold=None, min_length=50):
    """Derive centerline from distance-to-bank field on irregular HECRAS mesh.

    Strategy:
    1. Find wetted cells with valid distances
    2. Select ridge (cells above threshold)
    3. Order ridge points by nearest-neighbor traversal
    4. Smooth and return LineString if long enough
    """
    coords = np.asarray(coords, dtype=float)
    distances = np.asarray(distances, dtype=float)
    wetted_mask = np.asarray(wetted_mask, dtype=bool)
    valid_mask = wetted_mask & np.isfinite(distances)
    valid_coords = coords[valid_mask]
    valid_distances = distances[valid_mask]
    if len(valid_coords) == 0:
        return None
    if min_distance_threshold is None:
        min_distance_threshold = np.percentile(valid_distances, 75)
    ridge_mask = valid_distances >= min_distance_threshold
    ridge_coords = valid_coords[ridge_mask]
    ridge_distances = valid_distances[ridge_mask]
    if len(ridge_coords) == 0:
        return None
    if len(ridge_coords) < 10:
        return None

    try:
        tree = cKDTree(ridge_coords)
    except Exception:
        return None

    # build ordered path by greedy nearest neighbor
    start_idx = int(np.argmax(ridge_distances))
    order = [start_idx]
    remaining = set(range(len(ridge_coords))) - {start_idx}
    while remaining:
        cur = order[-1]
        dists, idxs = tree.query(ridge_coords[cur], k=len(ridge_coords))
        next_idx = None
        for cand in np.atleast_1d(idxs):
            if int(cand) in remaining:
                next_idx = int(cand)
                break
        if next_idx is None:
            break
        order.append(next_idx)
        remaining.remove(next_idx)

    ordered_coords = ridge_coords[order]
    if len(ordered_coords) > 5:
        sigma = max(1, len(ordered_coords) // 20)
        smoothed_x = gaussian_filter1d(ordered_coords[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(ordered_coords[:, 1], sigma=sigma)
        ordered_coords = np.column_stack((smoothed_x, smoothed_y))

    centerline = LineString(ordered_coords)
    if centerline.length < min_length:
        return None
    return centerline


def extract_centerline_fast_hecras(plan_coords, depth_arr, depth_threshold=0.05, sample_fraction=0.1, min_length=50):
    """Fast centerline extraction by sampling wetted cells and ordering by PCA projection.

    Returns a LineString or None.
    """
    coords = np.asarray(plan_coords, dtype=float)
    depth = np.asarray(depth_arr, dtype=float)
    wetted_mask = depth > depth_threshold
    coords_wet = coords[wetted_mask]
    if len(coords_wet) < 10:
        return None
    n_sample = max(50, int(len(coords_wet) * sample_fraction))
    idx = np.random.choice(len(coords_wet), size=n_sample, replace=False)
    sample = coords_wet[idx]
    # PCA
    mean = sample.mean(axis=0)
    X = sample - mean
    cov = np.dot(X.T, X) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, np.argmax(vals)]
    proj = (sample - mean).dot(principal)
    order = np.argsort(proj)
    ordered = sample[order]
    if len(ordered) > 5:
        from scipy.ndimage import gaussian_filter1d as _gf
        sigma = max(1, len(ordered) // 20)
        ox = _gf(ordered[:, 0], sigma=sigma)
        oy = _gf(ordered[:, 1], sigma=sigma)
        ordered = np.column_stack((ox, oy))
    line = LineString(ordered)
    if line.length < min_length:
        return None
    return line
"""Centerline extraction helpers ported from legacy hecras_helpers.

This module provides `derive_centerline_from_hecras_distance` which finds
ridge points (high distance-to-bank), orders them by nearest neighbor, and
returns a smoothed `shapely.geometry.LineString` when the extracted centerline
meets a minimum length threshold.

The implementation uses `emergent.fish_passage.utils.safe_build_kdtree` for
KDTree construction consistent with package conventions.
"""
from typing import Optional
import numpy as np
from shapely.geometry import LineString

from emergent.fish_passage.utils import safe_build_kdtree


def derive_centerline_from_hecras_distance(coords: np.ndarray,
                                           distances: np.ndarray,
                                           wetted_mask: np.ndarray,
                                           min_distance_threshold: Optional[float]=None,
                                           min_length: float=50.0) -> Optional[LineString]:
    """Derive a centerline from distance-to-bank ridge points.

    Parameters
    - coords: (N,2) array of point coordinates
    - distances: (N,) array of distance-to-bank values (may contain NaN)
    - wetted_mask: boolean mask of length N indicating wetted cells
    - min_distance_threshold: if None, use 75th percentile of valid distances
    - min_length: minimum centerline length (in same units as coords) to accept

    Returns a `LineString` or `None` if no valid centerline found.
    """
    from scipy.ndimage import gaussian_filter1d

    valid_mask = wetted_mask & np.isfinite(distances)
    valid_coords = coords[valid_mask]
    valid_distances = distances[valid_mask]
    if len(valid_coords) == 0:
        return None
    if min_distance_threshold is None:
        min_distance_threshold = float(np.percentile(valid_distances, 75))
    ridge_mask = valid_distances >= min_distance_threshold
    ridge_coords = valid_coords[ridge_mask]
    ridge_distances = valid_distances[ridge_mask]
    if len(ridge_coords) == 0:
        return None

    # Build KDTree via package utility
    ridge_tree = safe_build_kdtree(ridge_coords, name='hecras_ridge_tree')
    if ridge_tree is None or len(ridge_coords) == 0:
        return None

    # Greedy nearest-neighbor ordering starting from the largest ridge value
    start_idx = int(np.argmax(ridge_distances))
    ordered_indices = [start_idx]
    remaining = set(range(len(ridge_coords))) - {start_idx}
    current_idx = start_idx
    # Use cKDTree for ordering
    try:
        from scipy.spatial import cKDTree
        kdt = cKDTree(ridge_coords)
    except Exception:
        # If cKDTree unavailable, fall back to the safe tree's query
        kdt = ridge_tree

    while remaining:
        current_pt = ridge_coords[current_idx]
        # query neighbors up to remaining size
        k = min(len(ridge_coords), len(remaining) + 1)
        dists, indices = kdt.query(current_pt, k=k)
        next_idx = None
        for idx in np.atleast_1d(indices):
            if int(idx) in remaining:
                next_idx = int(idx)
                break
        if next_idx is None:
            break
        ordered_indices.append(next_idx)
        remaining.remove(next_idx)
        current_idx = next_idx

    ordered_coords = ridge_coords[ordered_indices]
    if len(ordered_coords) > 5:
        sigma = max(1, len(ordered_coords) // 20)
        smoothed_x = gaussian_filter1d(ordered_coords[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(ordered_coords[:, 1], sigma=sigma)
        ordered_coords = np.column_stack((smoothed_x, smoothed_y))

    centerline = LineString(ordered_coords)
    if centerline.length < min_length:
        return None
    return centerline


def extract_centerline_fast(coords: np.ndarray, depths: np.ndarray, depth_threshold: float = 0.05, sample_fraction: float = 0.1, min_length: float = 50.0):
    """Array-based fast centerline extraction (PCA + smoothing).

    This mirrors the legacy `extract_centerline_fast_hecras` but operates on
    in-memory `coords` and `depths` arrays to ease unit testing.
    """
    wetted_mask = depths > depth_threshold
    coords_wet = coords[wetted_mask]
    if len(coords_wet) < 10:
        return None
    n_sample = max(50, int(len(coords_wet) * sample_fraction))
    idx = np.random.choice(len(coords_wet), size=n_sample, replace=False)
    sample = coords_wet[idx]
    mean = sample.mean(axis=0)
    cov = np.cov((sample - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    proj = (sample - mean).dot(principal)
    order = np.argsort(proj)
    ordered = sample[order]
    from scipy.ndimage import gaussian_filter1d
    if len(ordered) > 5:
        sigma = max(1, len(ordered) // 50)
        sx = gaussian_filter1d(ordered[:, 0], sigma=sigma)
        sy = gaussian_filter1d(ordered[:, 1], sigma=sigma)
        ordered = np.column_stack((sx, sy))
    centerline = LineString(ordered)
    if centerline.length < min_length:
        return None
    return centerline


def extract_centerline_fast_hecras(plan_path: str, depth_threshold: float = 0.05, sample_fraction: float = 0.1, min_length: float = 50.0):
    """File-backed wrapper that reads coords/depths from a HECRAS HDF5 plan and calls `extract_centerline_fast`.
    """
    import h5py
    with h5py.File(plan_path, 'r') as hdf:
        coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])
        # depth dataset can be located at a few common paths
        depth_candidates = [
            'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth',
            'Results/Results_0001/Cell Hydraulic Depth/Values',
            'Results/Results_0001/Cell Hydraulic Depth',
            'Cell Hydraulic Depth',
        ]
        ds = None
        for c in depth_candidates:
            if c in hdf:
                ds = hdf[c]
                break
        if ds is None:
            # For minimal plans used in smoke tests, absence of depth is acceptable
            # and we return None to indicate no centerline could be derived.
            return None
        if getattr(ds, 'ndim', 0) > 0 and ds.shape[0] > 1:
            depth = np.array(ds[0])
        else:
            depth = np.array(ds[0])
    return extract_centerline_fast(coords, depth, depth_threshold=depth_threshold, sample_fraction=sample_fraction, min_length=min_length)


def infer_wetted_perimeter_from_arrays(coords: np.ndarray, depth: np.ndarray, depth_threshold: float = 0.05, max_nodes: int = 5000, raster_fallback_resolution: float = 5.0, verbose: bool = False):
    """Infer wetted perimeter from arrays of coords and depth values.

    This function mirrors legacy `infer_wetted_perimeter_from_hecras` vector path
    and includes a raster fallback that rasterizes points to a coarse grid,
    polygonizes, and returns exterior coordinates.
    """
    from shapely.geometry import Polygon
    from scipy.spatial import cKDTree

    wetted_mask = depth > depth_threshold
    if not np.any(wetted_mask):
        return None
    # Vector approach: if HECRAS-style facepoints/perimeter mapping isn't available,
    # we attempt a Delaunay/perimeter via convex hull of wetted points if small.
    wetted_coords = coords[wetted_mask]
    n = len(wetted_coords)
    if n == 0:
        return None
    if n <= max_nodes:
        # crude approach: compute convex hull of wetted points as perimeter proxy
        try:
            from shapely.geometry import MultiPoint
            hull = MultiPoint(wetted_coords).convex_hull
            if hull.is_empty:
                return None
            if isinstance(hull, Polygon):
                exterior = np.array(hull.exterior.coords)
                return exterior
            else:
                return np.array(hull.coords)
        except Exception:
            pass

    # Raster fallback: coarse rasterize points to grid and polygonize
    if verbose:
        print('Using raster fallback for wetted perimeter')
    # Estimate bounds and grid
    minx, miny = float(coords[:, 0].min()), float(coords[:, 1].min())
    maxx, maxy = float(coords[:, 0].max()), float(coords[:, 1].max())
    nx = max(1, int(np.ceil((maxx - minx) / raster_fallback_resolution)))
    ny = max(1, int(np.ceil((maxy - miny) / raster_fallback_resolution)))
    # Build occupancy grid
    cols = np.linspace(minx, maxx, nx)
    rows = np.linspace(miny, maxy, ny)
    col_grid, row_grid = np.meshgrid(cols, rows)
    grid_xy = np.column_stack((col_grid.flatten(), row_grid.flatten()))
    tree = cKDTree(coords)
    dists, idxs = tree.query(grid_xy, k=1)
    # assign grid cells as wetted if nearest point is wetted
    nearest_wetted = wetted_mask[idxs]
    grid_mask = nearest_wetted.reshape((ny, nx))
    # polygonize contiguous wetted regions using shapely via marching squares-like approach
    try:
        import shapely.geometry as geom
        from shapely.ops import unary_union
        polys = []
        for i in range(ny):
            for j in range(nx):
                if grid_mask[i, j]:
                    x0 = cols[j]
                    y0 = rows[i]
                    x1 = cols[j] + (cols[1]-cols[0]) if nx>1 else cols[j]
                    y1 = rows[i] + (rows[1]-rows[0]) if ny>1 else rows[i]
                    polys.append(geom.box(x0, y0, x1, y1))
        if not polys:
            return None
        u = unary_union(polys)
        # take largest polygon exterior
        if u.type == 'Polygon':
            exterior = np.array(u.exterior.coords)
            return exterior
        elif u.type == 'MultiPolygon':
            largest = max(u.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)
    except Exception:
        return None


def infer_wetted_perimeter_from_hecras(plan_path: str, depth_threshold: float = 0.05, max_nodes: int = 5000, raster_fallback_resolution: float = 5.0, verbose: bool = False, timestep: int = 0):
    """Read depth/coords from HECRAS plan and infer wetted perimeter.

    Returns a numpy array of perimeter coordinates (N,2) or None when no
    wetted perimeter can be inferred.
    """
    import h5py
    with h5py.File(plan_path, 'r') as hdf:
        if 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate' not in hdf:
            raise KeyError('HECRAS plan missing Cells Center Coordinate dataset')
        coords = np.asarray(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'])

        # search for depth dataset in common locations
        depth_candidates = [
            'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth',
            'Results/Results_0001/Cell Hydraulic Depth/Values',
            'Results/Results_0001/Cell Hydraulic Depth',
            'Cell Hydraulic Depth',
        ]
        ds = None
        for c in depth_candidates:
            if c in hdf:
                ds = hdf[c]
                break
        if ds is None:
            # no depth information — cannot infer wetted perimeter
            return None

        # read depth array (handle timeseries)
        if getattr(ds, 'ndim', 0) > 0 and ds.shape[0] > 1:
            depth = np.asarray(ds[min(timestep, ds.shape[0]-1)])
        else:
            depth = np.asarray(ds[0]) if getattr(ds, 'ndim', 0) > 0 else np.asarray(ds)

    res = infer_wetted_perimeter_from_arrays(coords, depth, depth_threshold=depth_threshold, max_nodes=max_nodes, raster_fallback_resolution=raster_fallback_resolution, verbose=verbose)
    return res



