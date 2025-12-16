"""behaviors

Pure-Pedantic, small behavior primitives used by higher-level controllers.

This module provides lightweight, well-tested pure-Python implementations
of a few simple behavioral primitives used by the simulation and tests:

- ``swim_upstream``: return per-agent unit vectors pointing toward an
  upstream direction.
- ``avoid_obstacle``: simple repulsion vectors from circular obstacles.
- ``school_with_neighbors``: cohesion/alignment/separation composite
  vector using a KDTree neighbor search.

Implementations are deliberately small and deterministic so unit tests
can rely on numeric parity with legacy behaviour while being easy to
read and maintain.
"""

from typing import Any, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
	"swim_upstream",
	"avoid_obstacle",
	"school_with_neighbors",
]


def swim_upstream(upstream_vector: Iterable[float], n_agents: int, strength: float = 1.0, mask: Optional[Iterable[bool]] = None) -> np.ndarray:
	"""Return desired per-agent vectors pointing upstream.

	Parameters
	- ``upstream_vector``: 2-element vector pointing upstream (x,y).
	- ``n_agents``: number of agents to produce outputs for.
	- ``strength``: scale applied to unit upstream vector.
	- ``mask``: optional boolean mask; False entries get zero vectors.

	Returns an (n_agents, 2) float array of vectors.
	"""
	uv = np.asarray(upstream_vector, dtype=float)
	if uv.shape != (2,):
		uv = np.asarray(uv).reshape(2,)
	norm = np.hypot(uv[0], uv[1])
	if norm == 0.0:
		unit = np.zeros(2, dtype=float)
	else:
		unit = uv / norm
	out = np.tile(unit * float(strength), (int(n_agents), 1))
	if mask is not None:
		m = np.asarray(mask, dtype=bool)
		if m.shape[0] != out.shape[0]:
			raise ValueError("mask length must equal n_agents")
		out[~m] = 0.0
	return out


def avoid_obstacle(positions: np.ndarray, obstacles: Iterable[Iterable[float]], influence: float = 1.0) -> np.ndarray:
	"""Compute repulsion vectors from circular obstacles.

	``positions`` is an (N,2) array. ``obstacles`` is an iterable of
	(x,y,radius) triples. ``influence`` is added to each obstacle radius
	to form the effective repulsion zone. The returned array is (N,2).
	"""
	pos = np.asarray(positions, dtype=float)
	if pos.ndim != 2 or pos.shape[1] != 2:
		raise ValueError("positions must be an (N,2) array")
	out = np.zeros_like(pos)
	# accept any iterable of triples; convert to array for vector ops
	obs_arr = np.asarray(list(obstacles), dtype=float)
	if obs_arr.size == 0:
		return out
	if obs_arr.ndim != 2 or obs_arr.shape[1] < 2:
		raise ValueError("obstacles must be iterable of (x,y[,radius])")
	# if radius missing, assume zero
	if obs_arr.shape[1] == 2:
		obs_arr = np.hstack((obs_arr, np.zeros((obs_arr.shape[0], 1), dtype=float)))
	for cx, cy, r in obs_arr:
		d = pos - np.array([cx, cy], dtype=float)
		dist = np.hypot(d[:, 0], d[:, 1])
		eff = float(r) + float(influence)
		# where inside influence zone, compute repulsion magnitude
		mask = dist < eff
		if not np.any(mask):
			continue
		# avoid divide-by-zero
		safe = dist[mask].copy()
		safe[safe == 0.0] = 1e-12
		repel = (eff - safe) / eff
		unitx = d[mask, 0] / safe
		unity = d[mask, 1] / safe
		out[mask, 0] += unitx * repel
		out[mask, 1] += unity * repel
	return out


def school_with_neighbors(positions: np.ndarray, headings: np.ndarray, body_lengths: Optional[np.ndarray] = None, behavioral_weights: Optional[Any] = None, alive_mask: Optional[np.ndarray] = None) -> np.ndarray:
	"""Return a composite vector per-agent from cohesion, alignment, and separation.

	This is a small, test-friendly implementation: for each agent we compute
	a cohesion vector (towards neighbor centroid), an alignment vector
	(difference to mean neighbor heading) and a separation vector (repulsion
	from very-close neighbors). Each term is scaled by weights from
	``behavioral_weights`` when available.
	"""
	pos = np.asarray(positions, dtype=float)
	headings = np.asarray(headings, dtype=float)
	N = pos.shape[0]
	if body_lengths is None:
		mean_BL = 1.0
	else:
		mean_BL = float(np.maximum(1.0, np.mean(np.asarray(body_lengths, dtype=float))))
	bw = behavioral_weights or type("W", (), {})()
	cohesion_r = float(getattr(bw, "cohesion_radius_relaxed", 2.0)) * mean_BL
	separation_r = float(getattr(bw, "separation_radius", 1.0)) * mean_BL
	cohesion_w = float(getattr(bw, "cohesion_weight", 1.0))
	alignment_w = float(getattr(bw, "alignment_weight", 1.0))
	separation_w = float(getattr(bw, "separation_weight", 1.0))

	if alive_mask is not None:
		alive = np.asarray(alive_mask, dtype=bool)
	else:
		alive = np.ones(N, dtype=bool)

	if N == 0:
		return np.zeros((0, 2), dtype=float)

	tree = cKDTree(pos)
	neighbor_lists = tree.query_ball_point(pos, r=cohesion_r)
	out = np.zeros((N, 2), dtype=float)
	for i in range(N):
		if not alive[i]:
			continue
		nbrs = [j for j in neighbor_lists[i] if j != i and alive[j]]
		if len(nbrs) == 0:
			continue
		# cohesion: vector towards centroid
		centroid = np.mean(pos[nbrs], axis=0)
		coh_vec = centroid - pos[i]
		# alignment: vector towards mean heading unit vector
		mean_sin = np.mean(np.sin(headings[nbrs]))
		mean_cos = np.mean(np.cos(headings[nbrs]))
		mean_heading = np.arctan2(mean_sin, mean_cos)
		align_vec = np.array([np.cos(mean_heading), np.sin(mean_heading)]) - np.array([np.cos(headings[i]), np.sin(headings[i])])
		# separation: repel from neighbors within separation_r
		sep_vec = np.zeros(2, dtype=float)
		for j in nbrs:
			d = pos[i] - pos[j]
			dist = np.hypot(d[0], d[1])
			if dist < separation_r and dist > 0.0:
				sep_vec += (d / dist) * (1.0 - dist / separation_r)
		total = cohesion_w * coh_vec + alignment_w * align_vec + separation_w * sep_vec
		out[i] = total
	return out

