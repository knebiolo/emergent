"""Utility helpers for fish_passage used across geometry and agents."""
from typing import Any
import numpy as np


def safe_build_kdtree(points: Any, name: str = 'KDTree'):
    """Build a scipy cKDTree defensively. Returns None on expected issues.

    - points: array-like
    - name: label for logging
    """
    try:
        if points is None:
            return None
        pts = np.asarray(points)
        if pts.size == 0:
            return None
        from scipy.spatial import cKDTree
        return cKDTree(pts)
    except (ValueError, TypeError, IndexError, AttributeError):
        return None


def standardize_shape(arr, target_shape=(5, 5), fill_value=np.nan):
    """Ensure array has `target_shape`, padding or trimming as needed.

    Useful for tests that expect fixed-size fixtures.
    """
    a = np.asarray(arr)
    th, tw = target_shape
    out = np.full((th, tw), fill_value, dtype=float)
    h = min(th, a.shape[0]) if a.ndim >= 1 else 0
    w = min(tw, a.shape[1]) if a.ndim >= 2 else 0
    if a.ndim == 2 and h > 0 and w > 0:
        out[:h, :w] = a[:h, :w]
    elif a.ndim == 1 and th * tw == a.size:
        out[:, :] = a.reshape((th, tw))
    return out
"""
utils.py

Small utility helpers moved from legacy code. This file contains a
defensive KDTree builder and a robust logging helper with minimal
defensive behavior. Keep implementations small and testable.

The public helpers added here:
- `safe_log_exception(msg, exc, **ctx)` : logs exceptions robustly
- `safe_build_kdtree(points, name='KDTree')` : returns a cKDTree or None

"""

from typing import Any, Optional
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)


def safe_log_exception(msg: str, exc: Exception, **ctx: Any) -> None:
	"""Log an exception robustly.

	Attempts to call `logger.exception`. If logging fails for any reason,
	falls back to writing a compact message to `sys.stderr`.
	"""
	try:
		if ctx:
			ctx_s = ' | '.join(f"{k}={v!r}" for k, v in ctx.items())
			logger.exception('%s | %s | %s', msg, exc, ctx_s)
		else:
			logger.exception('%s | %s', msg, exc)
	except Exception:
		# Minimal fallback: write a compact failure message to stderr.
		# Keep this tiny to preserve readability and avoid deep nested handlers.
		try:
			sys.stderr.write(f'LOGGING FAILURE: {msg} {exc}\n')
		except Exception:
			# Give up silently; don't allow logging fallback to raise.
			pass


def safe_build_kdtree(points: Any, name: str = 'KDTree') -> Optional[object]:
	"""Build a `scipy.spatial.cKDTree` for ``points`` defensively.

	Returns the tree instance or ``None`` for expected issues (empty input,
	None input, or simple type errors). Unexpected exceptions are re-raised
	after logging.
	"""
	try:
		if points is None:
			logger.debug('%s: points is None, not building tree', name)
			return None
		pts = np.asarray(points)
		if pts.size == 0:
			logger.debug('%s: points empty, not building tree', name)
			return None
		from scipy.spatial import cKDTree

		return cKDTree(pts)
	except (ValueError, TypeError, IndexError, AttributeError) as e:
		logger.exception('%s: failed to build cKDTree for provided points', name)
		return None
	except Exception:
		logger.exception('%s: unexpected error while building cKDTree; re-raising', name)
		raise


def get_inv_transform(sim, transform):
	"""Return cached inverse Affine for `transform` on `sim`.

	Caches by `id(transform)` on `sim._inv_transform_cache` to avoid repeated
	Affine inversion costs. If `sim` is None or not writable, falls back to
	returning `~transform` without caching.
	"""
	try:
		key = id(transform)
	except Exception:
		return ~transform
	cache = getattr(sim, '_inv_transform_cache', None)
	if cache is None:
		try:
			cache = {}
			setattr(sim, '_inv_transform_cache', cache)
		except Exception:
			return ~transform
	inv = cache.get(key)
	if inv is None:
		try:
			inv = ~transform
		except Exception:
			return ~transform
		cache[key] = inv
	return inv


def get_arr(use_gpu: bool = False):
	"""Return the array module: `cupy` when `use_gpu` and available, otherwise `numpy`.

	This helper centralizes runtime choice between `numpy` and `cupy` for optional GPU paths.
	"""
	if use_gpu:
		try:
			import cupy as cp
			return cp
		except Exception:
			# fall back to numpy if cupy unavailable
			pass
	return np


def precompute_pixel_indices(sim, mapping_keys: dict = None):
	"""Precompute and cache row/col indices for common raster transforms on `sim`.

	mapping_keys: optional dict mapping cache keys -> attribute name for transform
	  e.g. {'depth': 'depth_rast_transform', 'vel': 'vel_mag_rast_transform'}

	Stores result on `sim._pixel_index_cache` as dict[key] -> (rows, cols)
	where rows/cols are int32 arrays matching `sim.X`/`sim.Y` shape.
	"""
	if mapping_keys is None:
		mapping_keys = {
			'depth': 'depth_rast_transform',
			'vel': 'vel_mag_rast_transform',
			'vel_dir': 'vel_dir_rast_transform',
			'refugia': 'refugia_map_transform',
			'mental_map': 'mental_map_transform',
		}

	cache = {}
	X = getattr(sim, 'X', None)
	Y = getattr(sim, 'Y', None)
	if X is None or Y is None:
		raise RuntimeError('Simulation object must provide X and Y arrays for precompute')

	for key, attr in mapping_keys.items():
		transform = getattr(sim, attr, None)
		if transform is None:
			cache[key] = (np.full_like(X, -1, dtype=np.int32), np.full_like(Y, -1, dtype=np.int32))
			continue
		try:
			from emergent.fish_passage.geometry import geo_to_pixel_from_inv
			from emergent.fish_passage.utils import get_inv_transform
			inv = get_inv_transform(sim, transform)
			rows, cols = geo_to_pixel_from_inv(inv, X, Y)
		except (ValueError, TypeError, KeyError, IndexError, OSError) as e:
			try:
				# fallback to on-demand geo_to_pixel with transform
				from emergent.fish_passage.geometry import geo_to_pixel
				rows, cols = geo_to_pixel(transform, X, Y)
			except Exception:
				rows = np.full_like(X, -1, dtype=np.int32)
				cols = np.full_like(Y, -1, dtype=np.int32)
		cache[key] = (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))

	try:
		setattr(sim, '_pixel_index_cache', cache)
	except Exception:
		# best-effort: if sim not writable, return cache instead
		return cache
	return cache
