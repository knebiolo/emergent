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
