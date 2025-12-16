"""Utilities for `emergent.fish_passage`.

Expose small helpers used across the package.
"""

from .logging import get_logger, safe_log_exception, configure_console_handler, safe_build_kdtree
from .array_helpers import standardize_shape
from .agents import calculate_front_masks, determine_slices_from_vectors, determine_slices_from_headings

__all__ = [
	'get_logger', 'safe_log_exception', 'configure_console_handler', 'safe_build_kdtree', 'standardize_shape',
	'calculate_front_masks', 'determine_slices_from_vectors', 'determine_slices_from_headings'
]
