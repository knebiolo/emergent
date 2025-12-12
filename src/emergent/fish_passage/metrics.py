"""Agent-level metrics: schooling and drafting computations.

Provides numpy reference implementations and optional Numba-accelerated
variants may be added later. These implementations prioritize clarity and
testability over micro-optimizations.
"""
from typing import Dict, Any
import numpy as np
from scipy.spatial import cKDTree


def compute_schooling_metrics_biological(positions: np.ndarray, headings: np.ndarray, body_lengths: np.ndarray, behavioral_weights: Any, alive_mask=None) -> Dict[str, float]:
    """Compute simple schooling metrics used for behavioral feedback.

    Returns a dict with keys: 'cohesion_score', 'alignment_score',
    'separation_penalty', 'overall_schooling'. Scores are aggregated
    across agents (mean) and are bounded in reasonable ranges.
    """
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    body_lengths = np.asarray(body_lengths, dtype=float)

    if alive_mask is not None:
        positions = positions[alive_mask]
        headings = headings[alive_mask]
        body_lengths = body_lengths[alive_mask]

    N = len(positions)
    if N == 0:
        return {'cohesion_score': 0.0, 'alignment_score': 0.0, 'separation_penalty': 0.0, 'overall_schooling': 0.0}

    mean_BL = float(np.maximum(1.0, np.mean(body_lengths)))

    # Use a fixed neighborhood radius proportional to mean body length
    cohesion_radius = getattr(behavioral_weights, 'cohesion_radius_relaxed', 2.0) * mean_BL

    tree = cKDTree(positions)
    # query neighbors within radius for all agents
    neighbors = tree.query_ball_tree(tree, r=cohesion_radius)

    cohesion_scores = np.zeros(N, dtype=float)
    alignment_scores = np.zeros(N, dtype=float)
    separation_penalty = np.zeros(N, dtype=float)

    for i in range(N):
        nbrs = [j for j in neighbors[i] if j != i]
        if len(nbrs) == 0:
            cohesion_scores[i] = 0.0
            alignment_scores[i] = 0.0
            separation_penalty[i] = 0.0
            continue

        # COHESION: centroid distance
        centroid = positions[nbrs].mean(axis=0)
        dist_to_centroid = np.hypot(*(positions[i] - centroid))
        ideal_dist = getattr(behavioral_weights, 'separation_radius', 1.0) * mean_BL
        cohesion_scores[i] = float(np.exp(-0.5 * ((dist_to_centroid - ideal_dist) / (0.5 * mean_BL)) ** 2))

        # ALIGNMENT: circular mean of neighbor headings
        sin_sum = np.sum(np.sin(headings[nbrs]))
        cos_sum = np.sum(np.cos(headings[nbrs]))
        mean_heading = np.arctan2(sin_sum, cos_sum)
        heading_diff = np.arctan2(np.sin(headings[i] - mean_heading), np.cos(headings[i] - mean_heading))
        alignment_scores[i] = float(np.cos(heading_diff))

        # SEPARATION: penalty growing with number of very-close neighbors
        dists = np.hypot(*(positions[nbrs] - positions[i]).T)
        close_count = np.sum(dists < (0.5 * mean_BL))
        separation_penalty[i] = -float(close_count) / max(1, len(nbrs))

    cohesion_mean = float(np.clip(np.mean(cohesion_scores), 0.0, 1.0))
    alignment_mean = float(np.clip(np.mean(alignment_scores), -1.0, 1.0))
    separation_mean = float(np.clip(np.mean(separation_penalty), -1.0, 0.0))

    # overall_schooling: simple aggregation (weights can be tuned)
    overall = 0.5 * cohesion_mean + 0.4 * ((alignment_mean + 1.0) / 2.0) + 0.1 * (1.0 + separation_mean)
    overall = float(np.clip(overall, 0.0, 1.0))

    return {
        'cohesion_score': cohesion_mean,
        'alignment_score': alignment_mean,
        'separation_penalty': separation_mean,
        'overall_schooling': overall
    }


def compute_drafting_benefits(positions: np.ndarray, headings: np.ndarray, velocities: np.ndarray, body_lengths: np.ndarray, behavioral_weights: Any, alive_mask=None) -> np.ndarray:
    """Compute per-agent drag reduction factors based on drafting geometry.

    Returns an array with values in [0.0, 1.0] representing fractional
    drag reduction to apply (e.g., 0.15 for 15% reduction).
    """
    positions = np.asarray(positions, dtype=float)
    headings = np.asarray(headings, dtype=float)
    body_lengths = np.asarray(body_lengths, dtype=float)

    if alive_mask is not None:
        positions = positions[alive_mask]
        headings = headings[alive_mask]
        body_lengths = body_lengths[alive_mask]

    N = len(positions)
    if N == 0:
        return np.zeros(0, dtype=float)

    mean_BL = float(np.maximum(1.0, np.mean(body_lengths)))
    angle_tol_deg = getattr(behavioral_weights, 'drafting_angle_tolerance', 30.0)
    angle_tol = np.deg2rad(angle_tol_deg)

    # Build KDTree and query neighbors within a drafting distance (2 BL)
    tree = cKDTree(positions)
    neighbors = tree.query_ball_tree(tree, r=2.0 * mean_BL)

    reductions = np.zeros(N, dtype=float)
    for i in range(N):
        nbrs = [j for j in neighbors[i] if j != i]
        if not nbrs:
            reductions[i] = 0.0
            continue
        count_ahead = 0
        for j in nbrs:
            # vector from i to j in agent i frame
            vx, vy = positions[j] - positions[i]
            ang = np.arctan2(vy, vx)
            # relative angle between heading and vector to neighbor
            rel = np.arctan2(np.sin(ang - headings[i]), np.cos(ang - headings[i]))
            if abs(rel) < angle_tol:
                count_ahead += 1
        if count_ahead == 0:
            reductions[i] = 0.0
        elif count_ahead == 1:
            reductions[i] = getattr(behavioral_weights, 'drag_reduction_single', 0.15)
        else:
            reductions[i] = getattr(behavioral_weights, 'drag_reduction_dual', 0.25)

    return reductions
"""
metrics.py

Preamble/Module plan for metrics computation and aggregation (moved to fish_passage).

Responsibilities (planned):
- Compute schooling metrics (cohesion, alignment, separation), energy statistics, passage success rates.
- Provide streaming aggregators for per-timestep and per-episode statistics.
- Export helpers to format metrics for viewer UI and for logging/CSV outputs.

Notes:
- Implement unit-tested pure functions for core calculations; avoid side-effects.
"""
