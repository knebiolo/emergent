"""Utilities for `emergent.fish_passage`.

Expose small helpers used across the package.
"""

from .logging import get_logger, safe_log_exception, configure_console_handler, safe_build_kdtree
from .array_helpers import standardize_shape

__all__ = [
	'get_logger', 'safe_log_exception', 'configure_console_handler', 'safe_build_kdtree', 'standardize_shape'
]
