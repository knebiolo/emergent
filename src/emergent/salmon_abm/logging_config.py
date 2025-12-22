"""Minimal logging configuration helper for the salmon_abm package.

This module exposes a single helper `configure_logging` that sets up a
basic handler and format. Callers may import and override as desired.
"""
import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Configure module-level logging.

    - `level`: logging level to set (default INFO).
    - `fmt`: optional format string; if None a sensible default is used.
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    # Only configure basic logging if the root logger has no handlers.
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=fmt)
    else:
        # If handlers exist, ensure the level is at least what caller requested.
        root.setLevel(level)


__all__ = ["configure_logging"]
