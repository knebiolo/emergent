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
