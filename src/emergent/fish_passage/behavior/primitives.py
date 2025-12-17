"""Behavior primitives for fish_passage.

These are small, testable functions that compute per-agent desired vectors
based on neighborhood geometry. They are pure NumPy and can be wrapped by
numba later if needed.
"""
from typing import Tuple
import numpy as np


def schooling_vector(positions: np.ndarray, radii: float, ideal_dist: float) -> np.ndarray:
    """Compute a simple schooling attraction/repulsion vector per agent.

    positions: (n_agents, 2)
    radii: scalar or per-agent float (ignored for now)
    ideal_dist: desired inter-agent distance; agents closer than ideal_dist
                will be repelled, farther will be attracted.

    Returns: (n_agents, 2) vector of desired displacement directions (not normalized).
    """
    n = positions.shape[0]
    out = np.zeros_like(positions)
    for i in range(n):
        diffs = positions - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        # exclude self
        mask = dists > 0
        if not np.any(mask):
            continue
        rel = diffs[mask]
        dist_vals = dists[mask]
        # compute attraction/repulsion strength
        strength = (dist_vals - ideal_dist) / (ideal_dist + 1e-6)
        vec = np.sum(rel * strength[:, None], axis=0)
        out[i] = vec
    return out


def collision_avoidance_vector(positions: np.ndarray, min_sep: float) -> np.ndarray:
    """Compute repulsive vectors to avoid collisions within `min_sep`.

    Returns (n_agents, 2) repulsion vectors (sum of inverse-distance weighted)
    """
    n = positions.shape[0]
    out = np.zeros_like(positions)
    for i in range(n):
        diffs = positions[i] - positions
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists > 0) & (dists < min_sep)
        if not np.any(mask):
            continue
        rel = diffs[mask]
        dist_vals = dists[mask]
        weights = (min_sep - dist_vals) / (dist_vals + 1e-6)
        vec = np.sum(rel * weights[:, None], axis=0)
        out[i] = vec
    return out
