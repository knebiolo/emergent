"""Core utilities for fish passage module.

This file demonstrates the project's coding guidelines:
- small focused functions
- bounded loops with explicit max iterations
- startup-only allocation example
- minimal defensive handling and assertions
"""

from typing import List, Tuple
from types import SimpleNamespace
from typing import Optional
import importlib

# Example: startup-only allocation
_INITIAL_BUFFER_SIZE = 1024
_buffer = bytearray(_INITIAL_BUFFER_SIZE)


def compute_travel_times(distances: List[float], speed: float) -> List[float]:
    """Compute travel times for fish given distances and a constant speed.

    Args:
        distances: list of non-negative distances (meters).
        speed: positive speed (m/s).

    Returns:
        list of travel times in seconds, same length as `distances`.

    Notes:
    - Loop is bounded by the length of `distances` which is validated and must be <= 10000.
    - Function kept short and side-effect free.
    """
    assert speed > 0.0, "speed must be positive"
    n = len(distances)
    assert n <= 10000, "distances length must be <= 10000"

    times: List[float] = [0.0] * n
    # Bounded loop: iterates exactly n times; n is statically constrained by assertions
    for i in range(n):
        d = distances[i]
        assert d >= 0.0, "distance must be non-negative"
        times[i] = d / speed
    return times


def summarize_times(times: List[float]) -> Tuple[float, float]:
    """Return (min, max) of times. Raises AssertionError for empty input."""
    assert len(times) > 0, "times must be non-empty"
    mn = times[0]
    mx = times[0]
    # bounded loop
    for t in times:
        if t < mn:
            mn = t
        if t > mx:
            mx = t
    return mn, mx


def get_arr(use_cupy: Optional[bool] = None):
    """Return the array module to use: numpy by default, cupy if available and requested.

    Args:
        use_cupy: If True, prefer cupy. If False, force numpy. If None, autodetect based on
            whether cupy is importable.

    Returns:
        module: the array module (numpy or cupy-like API).
    """
    if use_cupy is False:
        import numpy as _np
        return _np
    try:
        if use_cupy is True:
            cupy = importlib.import_module("cupy")
            return cupy
        # autodetect
        cupy = importlib.import_module("cupy")
        return cupy
    except Exception:
        import numpy as _np
        return _np


class Simulation:
    """Minimal Simulation stub to act as an interim home for the sockeye monolith.

    This placeholder provides only the attributes and methods needed for import-time
    compatibility during the migration. Full behavior will be implemented by
    decomposing the original `sockeye.simulation` into smaller pieces.
    """
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.hdf5 = None

    def initialize(self):
        """Initialize simulation resources (stub)."""
        return True

    def step(self):
        """Advance simulation by one timestep (stub)."""
        raise NotImplementedError("Simulation.step is a migration stub")


