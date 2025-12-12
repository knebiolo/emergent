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

	# Build KDTree and neighbor lists (use query_ball_point for per-agent neighbors)
	from scipy.spatial import cKDTree as _cKDTree
	tree = _cKDTree(positions)
	neighbor_lists = tree.query_ball_point(positions, r=cohesion_radius)

	# Preallocate arrays
	cohesion_scores = np.zeros(N, dtype=float)
	alignment_scores = np.full(N, -0.5, dtype=float)
	separation_penalty = np.zeros(N, dtype=float)

	# Vectorized nearest neighbor separation penalty (keep previous behaviour)
	if N > 1:
		nearest_dists, nearest_indices = tree.query(positions, k=2)
		nearest_dists = nearest_dists[:, 1]
		mean_BL = float(np.mean(body_lengths))
		sep_mask = nearest_dists < mean_BL
		separation_penalty[sep_mask] = -(mean_BL - nearest_dists[sep_mask]) / mean_BL

		# Prepare flattened neighbor arrays for loop APIs
		neighbor_data = []
		neighbor_offsets = [0]
		for nbrs in neighbor_lists:
			neighbor_data.extend(nbrs)
			neighbor_offsets.append(len(neighbor_data))
		neighbor_data = np.array(neighbor_data, dtype=np.int32)
		neighbor_offsets = np.array(neighbor_offsets, dtype=np.int32)

		# Ideal distance for cohesion (legacy heuristic)
		ideal_dist = 2.0 * mean_BL * (1 - 0.5 * float(getattr(behavioral_weights, 'threat_level', 0.0)))

		# Call the ported loop (numba or python fallback)
		_compute_schooling_loop(neighbor_data, neighbor_offsets, positions, headings, ideal_dist, mean_BL,
				cohesion_scores, alignment_scores, N)

	cohesion = float(np.mean(cohesion_scores))
	alignment = float(np.mean(alignment_scores))
	separation = float(np.mean(separation_penalty))
	overall = cohesion + alignment + separation
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

	# Build KDTree and neighbor lists
	tree = cKDTree(positions)
	neighbor_lists = tree.query_ball_point(positions, r=forward_radius)

	reductions = np.zeros(N, dtype=float)

	# Prepare flattened neighbor arrays
	neighbor_data = []
	neighbor_offsets = [0]
	for nbrs in neighbor_lists:
		neighbor_data.extend(nbrs)
		neighbor_offsets.append(len(neighbor_data))
	neighbor_data = np.array(neighbor_data, dtype=np.int32)
	neighbor_offsets = np.array(neighbor_offsets, dtype=np.int32)

	drag_red_single = float(getattr(behavioral_weights, 'drag_reduction_single', 0.15))
	drag_red_dual = float(getattr(behavioral_weights, 'drag_reduction_dual', 0.25))

	# Dispatch to loop implementation (numba or python fallback)
	_compute_drafting_loop(neighbor_data, neighbor_offsets, positions, headings, angle_tol_rad,
				drag_red_single, drag_red_dual, reductions, N)

	return reductions


# Numba accelerated cores and compile helper
if _HAS_NUMBA:
	import math

	@njit(parallel=True, fastmath=True)
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
				dist = math.hypot(dx, dy)
				if dist <= cohesion_radius:
					cnt += 1
					# cohesion contribution
					tmp = 1.0 - dist / cohesion_radius
					if tmp > 0.0:
						coh_sum += tmp
					# alignment using scalar cos
					align_sum += math.cos(headings[j] - hi)
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


	@njit(parallel=True, fastmath=True)
	def _drafting_numba_core(positions, headings, forward_radius, angle_tol_rad, drag_red_single, drag_red_dual):
		N = positions.shape[0]
		reductions = np.zeros(N, dtype=np.float64)
		for i in prange(N):
			cnt = 0
			xi = positions[i, 0]
			yi = positions[i, 1]
			hi = headings[i]
			hvx = math.cos(hi)
			hvy = math.sin(hi)
			for j in range(N):
				if i == j:
					continue
				dx = positions[j, 0] - xi
				dy = positions[j, 1] - yi
				dist = math.hypot(dx, dy)
				if dist == 0.0 or dist > forward_radius:
					continue
				invd = 1.0 / dist
				dot = hvx * (dx * invd) + hvy * (dy * invd)
				if dot > 1.0:
					dot = 1.0
				if dot < -1.0:
					dot = -1.0
				angle = math.acos(dot)
				if angle <= angle_tol_rad:
					cnt += 1
			if cnt == 1:
				reductions[i] = drag_red_single
			elif cnt > 1:
				reductions[i] = drag_red_dual
		return reductions


	def compile_numba_kernels():
		"""Force Numba to compile the inner kernels using tiny dummy inputs.

		Call this once at startup to pay the JIT cost early (tests/benchmarks).
		"""
		# small deterministic arrays
		pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
		headings = np.array([0.0, 0.0], dtype=np.float64)
		# call once to trigger compilation
		_ = _schooling_numba_core(pos, headings, 2.0, 1.0)
		_ = _drafting_numba_core(pos, headings, 2.0, 0.5, 0.15, 0.25)


# Fallback / portable loop APIs that accept flattened neighbor arrays
def _compute_schooling_loop(neighbor_data, neighbor_offsets, positions, headings, ideal_dist, mean_BL,
							cohesion_scores, alignment_scores, N):
	"""Portable schooling loop. If Numba is available, the numba-jitted
	version will be used by assigning to the same name at import time.
	"""
	if _HAS_NUMBA:
		# call the compiled version present in legacy sockeye (if available)
		try:
			return _compute_schooling_loop_numba(neighbor_data, neighbor_offsets, positions, headings, ideal_dist, mean_BL,
												 cohesion_scores, alignment_scores, N)
		except NameError:
			pass

	# Pure Python fallback loop
	for i in range(N):
		start = neighbor_offsets[i]
		end = neighbor_offsets[i + 1]
		n_neighbors = 0
		for idx in range(start, end):
			if neighbor_data[idx] != i:
				n_neighbors += 1
		if n_neighbors == 0:
			continue
		# centroid
		centroid_x = 0.0
		centroid_y = 0.0
		for idx in range(start, end):
			j = neighbor_data[idx]
			if j != i:
				centroid_x += positions[j, 0]
				centroid_y += positions[j, 1]
		centroid_x /= n_neighbors
		centroid_y /= n_neighbors
		dx = positions[i, 0] - centroid_x
		dy = positions[i, 1] - centroid_y
		dist_to_centroid = (dx * dx + dy * dy) ** 0.5
		cohesion_scores[i] = np.exp(-0.5 * ((dist_to_centroid - ideal_dist) / (0.5 * mean_BL)) ** 2)
		# alignment
		sin_sum = 0.0
		cos_sum = 0.0
		for idx in range(start, end):
			j = neighbor_data[idx]
			if j != i:
				sin_sum += np.sin(headings[j])
				cos_sum += np.cos(headings[j])
		sin_mean = sin_sum / n_neighbors
		cos_mean = cos_sum / n_neighbors
		mean_heading = np.arctan2(sin_mean, cos_mean)
		heading_diff = np.arctan2(np.sin(headings[i] - mean_heading), np.cos(headings[i] - mean_heading))
		alignment_scores[i] = np.cos(heading_diff)


def _compute_drafting_loop(neighbor_data, neighbor_offsets, positions, headings, angle_tol_rad,
						   drag_reduction_single, drag_reduction_dual, drag_reductions, N):
	if _HAS_NUMBA:
		try:
			return _compute_drafting_loop_numba(neighbor_data, neighbor_offsets, positions, headings, angle_tol_rad,
												 drag_reduction_single, drag_reduction_dual, drag_reductions, N)
		except NameError:
			pass

	for i in range(N):
		start = neighbor_offsets[i]
		end = neighbor_offsets[i + 1]
		left_count = 0
		right_count = 0
		total_ahead = 0
		for idx in range(start, end):
			j = neighbor_data[idx]
			if j == i:
				continue
			dx = positions[j, 0] - positions[i, 0]
			dy = positions[j, 1] - positions[i, 1]
			angle_to_neighbor = np.arctan2(dy, dx)
			angle_diff = np.arctan2(np.sin(angle_to_neighbor - headings[i]), np.cos(angle_to_neighbor - headings[i]))
			if abs(angle_diff) < angle_tol_rad:
				total_ahead += 1
				if angle_diff < 0:
					left_count += 1
				else:
					right_count += 1
		if total_ahead == 1:
			drag_reductions[i] = drag_reduction_single
		elif total_ahead >= 2:
			if left_count > 0 and right_count > 0:
				drag_reductions[i] = drag_reduction_dual
			else:
				drag_reductions[i] = drag_reduction_single

# If Numba is available, expose compiled names expected by sockeye
if _HAS_NUMBA:
	try:
		# attempt to compile equivalents using numba.jit similarly to sockeye

		@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
		def _compute_schooling_loop_numba(neighbor_data, neighbor_offsets, positions, headings, ideal_dist, mean_BL,
										 cohesion_scores, alignment_scores, N):
			for i in numba.prange(N):
				start = neighbor_offsets[i]
				end = neighbor_offsets[i + 1]
				n_neighbors = 0
				for idx in range(start, end):
					if neighbor_data[idx] != i:
						n_neighbors += 1
				if n_neighbors > 0:
					centroid_x = 0.0
					centroid_y = 0.0
					for idx in range(start, end):
						j = neighbor_data[idx]
						if j != i:
							centroid_x += positions[j, 0]
							centroid_y += positions[j, 1]
					centroid_x /= n_neighbors
					centroid_y /= n_neighbors
					dx = positions[i, 0] - centroid_x
					dy = positions[i, 1] - centroid_y
					dist_to_centroid = (dx * dx + dy * dy) ** 0.5
					cohesion_scores[i] = math.exp(-0.5 * ((dist_to_centroid - ideal_dist) / (0.5 * mean_BL)) ** 2)
					sin_sum = 0.0
					cos_sum = 0.0
					for idx in range(start, end):
						j = neighbor_data[idx]
						if j != i:
							sin_sum += math.sin(headings[j])
							cos_sum += math.cos(headings[j])
					sin_mean = sin_sum / n_neighbors
					cos_mean = cos_sum / n_neighbors
					mean_heading = math.atan2(sin_mean, cos_mean)
					heading_diff = math.atan2(math.sin(headings[i] - mean_heading), math.cos(headings[i] - mean_heading))
					alignment_scores[i] = math.cos(heading_diff)


		@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
		def _compute_drafting_loop_numba(neighbor_data, neighbor_offsets, positions, headings, angle_tol_rad,
										 drag_reduction_single, drag_reduction_dual, drag_reductions, N):
			for i in numba.prange(N):
				start = neighbor_offsets[i]
				end = neighbor_offsets[i + 1]
				left_count = 0
				right_count = 0
				total_ahead = 0
				for idx in range(start, end):
					j = neighbor_data[idx]
					if j == i:
						continue
					dx = positions[j, 0] - positions[i, 0]
					dy = positions[j, 1] - positions[i, 1]
					angle_to_neighbor = math.atan2(dy, dx)
					angle_diff = math.atan2(math.sin(angle_to_neighbor - headings[i]), math.cos(angle_to_neighbor - headings[i]))
					if math.fabs(angle_diff) < angle_tol_rad:
						total_ahead += 1
						if angle_diff < 0:
							left_count += 1
						else:
							right_count += 1
				if total_ahead == 1:
					drag_reductions[i] = drag_reduction_single
				elif total_ahead >= 2:
					if left_count > 0 and right_count > 0:
						drag_reductions[i] = drag_reduction_dual
					else:
						drag_reductions[i] = drag_reduction_single
	except Exception:
		# If compiling numba helpers fails during import, fall back silently to Python versions
		pass

