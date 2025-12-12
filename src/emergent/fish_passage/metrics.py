"""Clean agent metrics: schooling and drafting.

This module provides clear numpy-only reference implementations and
optional Numba-accelerated inner loops when Numba is available.

The file is intentionally minimal and contains no duplicated blocks
so imports and names are not shadowed.
"""
from typing import Dict, Any

import numpy as np
from scipy.spatial import cKDTree


# Optional Numba accelerated inner kernels
try:
	import numba  # type: ignore
	from numba import njit, prange  # type: ignore
	_HAS_NUMBA = True
except Exception:
	_HAS_NUMBA = False


def compute_schooling_metrics_biological(positions: np.ndarray, headings: np.ndarray, body_lengths: np.ndarray, behavioral_weights: Any, alive_mask=None) -> Dict[str, float]:
	"""Compute cohesion, alignment and separation metrics for a group.

	Parameters mirror tests in src/emergent/fish_passage/tests/test_metrics.py.
	Returns a dict with keys: cohesion_score, alignment_score,
	separation_penalty, overall_schooling.
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
	cohesion_radius = float(getattr(behavioral_weights, 'cohesion_radius_relaxed', 2.0)) * mean_BL
	separation_radius = float(getattr(behavioral_weights, 'separation_radius', 1.0)) * mean_BL

	if _HAS_NUMBA:
		coh, align, sep, overall = _schooling_numba_core(positions, headings, cohesion_radius, separation_radius)
		return {'cohesion_score': float(coh), 'alignment_score': float(align), 'separation_penalty': float(sep), 'overall_schooling': float(overall)}

	tree = cKDTree(positions)
	neighbors = tree.query_ball_tree(tree, r=cohesion_radius)

	cohesion_scores = np.zeros(N, dtype=float)
	alignment_scores = np.zeros(N, dtype=float)
	separation_penalty = np.zeros(N, dtype=float)

	for i in range(N):
		nbrs = [j for j in neighbors[i] if j != i]
		if not nbrs:
			continue
		nbr_pos = positions[nbrs]
		dists = np.linalg.norm(nbr_pos - positions[i], axis=1)
		cohesion_scores[i] = float(np.mean(np.maximum(0.0, 1.0 - dists / cohesion_radius)))
		alignment_scores[i] = float(np.mean(np.cos(headings[nbrs] - headings[i])))
		close = dists < separation_radius
		if np.any(close):
			separation_penalty[i] = float(np.sum((separation_radius - dists[close]) / separation_radius) / len(dists))

	cohesion = float(np.mean(cohesion_scores))
	alignment = float(np.mean(alignment_scores))
	separation = float(np.mean(separation_penalty))
	overall = cohesion * 0.5 + (alignment + 1.0) * 0.25 - separation * 0.25
	return {'cohesion_score': cohesion, 'alignment_score': alignment, 'separation_penalty': separation, 'overall_schooling': overall}


def compute_drafting_benefits(positions: np.ndarray, headings: np.ndarray, velocities: np.ndarray, body_lengths: np.ndarray, behavioral_weights: Any, alive_mask=None) -> np.ndarray:
	"""Compute draft-related drag reductions per agent.

	Returns an array of reductions in [0,1].
	"""
	positions = np.asarray(positions, dtype=float)
	headings = np.asarray(headings, dtype=float)
	velocities = np.asarray(velocities, dtype=float)
	body_lengths = np.asarray(body_lengths, dtype=float)

	if alive_mask is not None:
		positions = positions[alive_mask]
		headings = headings[alive_mask]
		velocities = velocities[alive_mask]
		body_lengths = body_lengths[alive_mask]

	N = len(positions)
	if N == 0:
		return np.zeros(0, dtype=float)

	mean_BL = float(np.maximum(1.0, np.mean(body_lengths)))
	angle_tol_rad = np.deg2rad(float(getattr(behavioral_weights, 'drafting_angle_tolerance', 30.0)))
	forward_radius = float(getattr(behavioral_weights, 'drafting_forward_radius', 2.0)) * mean_BL

	if _HAS_NUMBA:
		drag_red_single = float(getattr(behavioral_weights, 'drag_reduction_single', 0.15))
		drag_red_dual = float(getattr(behavioral_weights, 'drag_reduction_dual', 0.25))
		return _drafting_numba_core(positions, headings, forward_radius, angle_tol_rad, drag_red_single, drag_red_dual)

	tree = cKDTree(positions)
	neighbors = tree.query_ball_tree(tree, r=forward_radius)

	reductions = np.zeros(N, dtype=float)
	for i in range(N):
		nbrs = [j for j in neighbors[i] if j != i]
		if not nbrs:
			continue
		count_ahead = 0
		for j in nbrs:
			vec = positions[j] - positions[i]
			dist = np.linalg.norm(vec)
			if dist == 0.0 or dist > forward_radius:
				continue
			heading_vec = np.array([np.cos(headings[i]), np.sin(headings[i])])
			cosang = np.dot(heading_vec, vec / dist)
			cosang = np.clip(cosang, -1.0, 1.0)
			angle = np.arccos(cosang)
			if angle <= angle_tol_rad:
				count_ahead += 1
		if count_ahead == 1:
			reductions[i] = float(getattr(behavioral_weights, 'drag_reduction_single', 0.15))
		elif count_ahead > 1:
			reductions[i] = float(getattr(behavioral_weights, 'drag_reduction_dual', 0.25))
	return reductions


# Numba accelerated cores
if _HAS_NUMBA:

	@njit(parallel=True)
	def _schooling_numba_core(positions, headings, cohesion_radius, separation_radius):
		N = positions.shape[0]
		cohesion_scores = np.zeros(N, dtype=np.float64)
		alignment_scores = np.zeros(N, dtype=np.float64)
		separation_penalty = np.zeros(N, dtype=np.float64)
		for i in prange(N):
			cnt = 0
			coh_sum = 0.0
			align_sum = 0.0
			sep_sum = 0.0
			xi = positions[i, 0]
			yi = positions[i, 1]
			hi = headings[i]
			for j in range(N):
				if i == j:
					continue
				dx = positions[j, 0] - xi
				dy = positions[j, 1] - yi
				dist = (dx * dx + dy * dy) ** 0.5
				if dist <= cohesion_radius:
					cnt += 1
					coh_sum += max(0.0, 1.0 - dist / cohesion_radius)
					align_sum += np.cos(headings[j] - hi)
					if dist < separation_radius:
						sep_sum += (separation_radius - dist) / separation_radius
			if cnt > 0:
				cohesion_scores[i] = coh_sum / cnt
				alignment_scores[i] = align_sum / cnt
				separation_penalty[i] = sep_sum / cnt
		coh = 0.0
		align = 0.0
		sep = 0.0
		for i in range(N):
			coh += cohesion_scores[i]
			align += alignment_scores[i]
			sep += separation_penalty[i]
		coh /= N
		align /= N
		sep /= N
		overall = coh * 0.5 + (align + 1.0) * 0.25 - sep * 0.25
		return coh, align, sep, overall


	@njit(parallel=True)
	def _drafting_numba_core(positions, headings, forward_radius, angle_tol_rad, drag_red_single, drag_red_dual):
		N = positions.shape[0]
		reductions = np.zeros(N, dtype=np.float64)
		for i in prange(N):
			cnt = 0
			xi = positions[i, 0]
			yi = positions[i, 1]
			hi = headings[i]
			hvx = np.cos(hi)
			hvy = np.sin(hi)
			for j in range(N):
				if i == j:
					continue
				dx = positions[j, 0] - xi
				dy = positions[j, 1] - yi
				dist = (dx * dx + dy * dy) ** 0.5
				if dist == 0.0 or dist > forward_radius:
					continue
				dot = (hvx * (dx / dist) + hvy * (dy / dist))
				if dot > 1.0:
					dot = 1.0
				if dot < -1.0:
					dot = -1.0
				angle = np.arccos(dot)
				if angle <= angle_tol_rad:
					cnt += 1
			if cnt == 1:
				reductions[i] = drag_red_single
			elif cnt > 1:
				reductions[i] = drag_red_dual
		return reductions

