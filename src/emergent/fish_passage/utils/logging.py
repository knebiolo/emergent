import logging
import sys
import traceback
from typing import Optional
import numpy as np


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger for the given name.

    If `level` is provided, set the logger level. The function will attach a
    StreamHandler only if the logger has no handlers to keep test output
    predictable.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    # Attach a simple StreamHandler if no handlers exist to make logs visible
    # in simple scripts while avoiding duplicate handlers in larger apps.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def safe_log_exception(logger: Optional[logging.Logger], message: Optional[str] = None, exc_info: bool = True) -> None:
    """Log an exception safely.

    - If `logger` is provided, attempts to call `logger.exception(message)`
      (if `exc_info` True) or `logger.error(message, exc_info=exc_info)`.
    - If logging fails or `logger` is falsy, falls back to writing a concise
    message and the traceback to stderr and never raises.
    """
    try:
        if logger:
            # Prefer logger.exception when exc_info requested
            if exc_info and hasattr(logger, "exception"):
                try:
                    logger.exception(message)
                    return
                except Exception:
                    # Fall through to the more generic error call below
                    pass
            # Generic error call
            logger.error(message if message is not None else "Exception occurred", exc_info=exc_info)
            return
    except Exception:
        # If the logger itself raises, fall back to stderr printing below
        pass

    # Fallback path: print to stderr with traceback
    try:
        if message:
            print(f"[safe_log_exception] {message}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    except Exception:
        # If even printing fails, give up silently (must not raise)
        try:
            sys.stderr.write("[safe_log_exception] failed to log exception\n")
        except Exception:
            pass


def configure_console_handler(logger: logging.Logger, fmt: Optional[str] = None, level: Optional[int] = None) -> None:
    """Configure a console handler for a logger (helper used in tests/scripts)."""
    if level is not None:
        logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt or "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def safe_build_kdtree(points, name: Optional[str] = 'KDTree'):
    """Build a cKDTree for `points` defensively.

    Returns the tree or None on expected runtime errors (logs the issue).
    Re-raises unexpected exceptions.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        try:
            logging.getLogger(__name__).exception('cKDTree backend not available')
        except Exception:
            pass
        return None

    logger = logging.getLogger(__name__)
    try:
        if points is None:
            logger.debug('%s: points is None, not building tree', name)
            return None
        pts = np.asarray(points)
        if pts.size == 0:
            logger.debug('%s: points empty, not building tree', name)
            return None
        return cKDTree(pts)
    except (ValueError, TypeError, IndexError, AttributeError):
        logger.exception('%s: failed to build cKDTree for provided points', name)
        return None
    except Exception:
        logger.exception('%s: unexpected error while building cKDTree; re-raising', name)
        raise
