"""Simulation package for the fish_passage migration.

This package will contain the refactored Simulation facade and submodules
(`state`, `io`, `kernels`, `step`) to replace the legacy `sockeye.simulation`.
"""

from .state import SimulationState

__all__ = ["SimulationState"]
