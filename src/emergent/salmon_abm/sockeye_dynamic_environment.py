"""Compatibility shim: provide a minimal import surface expected by package init.

This module attempts to re-export the canonical `sockeye` symbols when
available, otherwise provides a safe empty fallback so tools that import the
package for side-effects (like headless checks) don't fail at import-time.
"""
try:
    from .sockeye import *  # noqa: F401,F403
except Exception:
    # Provide minimal placeholders if sockeye is unavailable; real code
    # should import the canonical `sockeye` module directly.
    __all__ = []
