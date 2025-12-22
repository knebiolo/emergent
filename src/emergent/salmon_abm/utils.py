"""Utility helpers extracted from sockeye.py

This module contains small, pure helpers to centralize common functionality
used by the salmon ABM: coordinate transforms, array helpers, and simple
geometry utilities.
"""
import numpy as np


def geo_to_pixel(X, Y, transform):
    """Convert geographic coordinates (X,Y) to raster row/col indices.

    Parameters
    - X, Y: array-like coordinates
    - transform: affine transform with invertible operator (~transform)

    Returns
    - rows, cols: integer arrays of pixel indices
    """
    inv_transform = ~transform
    pixels = [inv_transform * (x, y) for x, y in zip(X, Y)]
    cols, rows = zip(*pixels)
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)
    return rows, cols


def pixel_to_geo(transform, rows, cols):
    """Convert raster row/col to geographic coordinates (x,y).

    Uses the rasterio Affine convention: x = c + a*(col+0.5), y = f + e*(row+0.5)
    """
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)
    return xs, ys


def standardize_shape(arr, target_shape=(5, 5), fill_value=np.nan):
    if arr.shape != target_shape:
        standardized_arr = np.full(target_shape, fill_value)
        standardized_arr[:arr.shape[0], :arr.shape[1]] = arr
        return standardized_arr
    return arr


def calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0):
    """Return masks indicating whether grid cells are in front of agents.

    headings : array-like of size (n_agents,) in radians
    x_coords, y_coords : arrays shaped (n_agents, H, W) giving cell coords
    agent_x, agent_y : arrays shaped (n_agents,) agent positions

    Returns array of shape (n_agents, H, W) with 1 in front, behind_value otherwise.
    """
    num_agents = len(headings)
    dx = np.cos(headings)[:, np.newaxis, np.newaxis]
    dy = np.sin(headings)[:, np.newaxis, np.newaxis]
    agent_x_expanded = agent_x[:, np.newaxis, np.newaxis]
    agent_y_expanded = agent_y[:, np.newaxis, np.newaxis]
    rel_x = x_coords - agent_x_expanded
    rel_y = y_coords - agent_y_expanded
    dot_product = dx * rel_x + dy * rel_y
    front_masks = (dot_product > 0).astype(int)
    front_masks[dot_product <= 0] = behind_value
    return front_masks


def determine_slices_from_vectors(vectors, num_slices=4):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    normalized_angles = np.mod(angles, 2*np.pi)
    slice_width = 2*np.pi / num_slices
    slice_indices = (normalized_angles // slice_width).astype(int)
    return slice_indices


def determine_slices_from_headings(headings, num_slices=4):
    normalized_headings = np.mod(headings, 2*np.pi)
    slice_width = 2*np.pi / num_slices
    slice_indices = (normalized_headings // slice_width).astype(int)
    return slice_indices


__all__ = [
    "geo_to_pixel",
    "pixel_to_geo",
    "standardize_shape",
    "calculate_front_masks",
    "determine_slices_from_vectors",
    "determine_slices_from_headings",
]
