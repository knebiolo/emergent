"""Perception and slicing helpers ported from legacy callers.

Functions are small, pure-numpy, and fully unit-tested.
"""
from typing import Tuple
import numpy as np


def calculate_front_masks(headings: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, agent_x: np.ndarray, agent_y: np.ndarray, behind_value: int = 0) -> np.ndarray:
    """Return boolean/int masks (N, H, W) marking cells in front of each agent.

    Parameters
    - headings: (N,) array of angles in radians (0 = +x direction)
    - x_coords, y_coords: arrays of shape (H, W) giving cell center coordinates
    - agent_x, agent_y: (N,) arrays of agent positions
    - behind_value: value to assign to cells behind the agent (default 0)

    Returns
    - front_masks: integer array shape (N, H, W) with 1 for front cells and behind_value for others
    """
    headings = np.asarray(headings, dtype=float)
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    agent_x = np.asarray(agent_x, dtype=float)
    agent_y = np.asarray(agent_y, dtype=float)

    if headings.ndim != 1:
        raise ValueError('headings must be 1-D array')
    N = headings.shape[0]

    # direction vectors (N,1,1)
    dx = np.cos(headings)[:, None, None]
    dy = np.sin(headings)[:, None, None]

    # expand agent coords to (N,1,1)
    ax = agent_x[:, None, None]
    ay = agent_y[:, None, None]

    # relative vectors from agent to each cell: broadcast to (N,H,W)
    rel_x = x_coords[None, :, :] - ax
    rel_y = y_coords[None, :, :] - ay

    dot = dx * rel_x + dy * rel_y
    front = (dot > 0)
    masks = np.where(front, 1, behind_value).astype(int)
    return masks


def determine_slices_from_vectors(vectors: np.ndarray, num_slices: int = 4) -> np.ndarray:
    """Assign each 2D vector to an angular slice index [0..num_slices-1].

    - vectors: (M,2) array of (vx, vy)
    - num_slices: number of equal-width angular bins
    """
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError('vectors must be shape (M,2)')
    angles = np.arctan2(v[:, 1], v[:, 0])
    normalized = np.mod(angles, 2 * np.pi)
    slice_width = 2 * np.pi / float(num_slices)
    idx = (normalized // slice_width).astype(int)
    return idx


def determine_slices_from_headings(headings: np.ndarray, num_slices: int = 4) -> np.ndarray:
    """Assign headings (radians) to slice indices."""
    h = np.asarray(headings, dtype=float)
    normalized = np.mod(h, 2 * np.pi)
    slice_width = 2 * np.pi / float(num_slices)
    idx = (normalized // slice_width).astype(int)
    return idx
