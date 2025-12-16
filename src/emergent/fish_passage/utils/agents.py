import numpy as np
from typing import Sequence


def calculate_front_masks(headings: Sequence[float], x_coords: np.ndarray, y_coords: np.ndarray, agent_x: Sequence[float], agent_y: Sequence[float], behind_value=0):
    """Return a (N, H, W) mask indicating which cells are in front of each agent.

    - `headings`: sequence of agent headings (radians)
    - `x_coords`, `y_coords`: arrays shaped (H, W) of cell center coordinates
    - `agent_x`, `agent_y`: sequences of agent positions (length N)
    """
    headings = np.asarray(headings, dtype=float)
    num_agents = len(headings)

    dx = np.cos(headings)[:, None, None]
    dy = np.sin(headings)[:, None, None]

    agent_x = np.asarray(agent_x)[:, None, None]
    agent_y = np.asarray(agent_y)[:, None, None]

    rel_x = x_coords[None, :, :] - agent_x
    rel_y = y_coords[None, :, :] - agent_y

    dot_product = dx * rel_x + dy * rel_y
    front_masks = (dot_product > 0).astype(int)
    front_masks[dot_product <= 0] = behind_value
    return front_masks


def determine_slices_from_vectors(vectors: np.ndarray, num_slices: int = 4):
    vecs = np.asarray(vectors, dtype=float)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    normalized = np.mod(angles, 2 * np.pi)
    slice_width = 2 * np.pi / float(num_slices)
    return (normalized // slice_width).astype(int)


def determine_slices_from_headings(headings: Sequence[float], num_slices: int = 4):
    h = np.asarray(headings, dtype=float)
    normalized = np.mod(h, 2 * np.pi)
    slice_width = 2 * np.pi / float(num_slices)
    return (normalized // slice_width).astype(int)
