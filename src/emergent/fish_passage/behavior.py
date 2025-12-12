"""Behavior-related small data containers.

Holds simple parameter containers used across metrics and RL code.
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class BehavioralWeights:
    """Container for behavioral weight parameters used by metrics.

    Defaults chosen to match legacy behavior used in tests.
    """
    cohesion_radius_relaxed: float = 2.0
    separation_radius: float = 1.0
    drafting_angle_tolerance: float = 30.0
    drafting_forward_radius: float = 2.0
    drag_reduction_single: float = 0.15
    drag_reduction_dual: float = 0.25

    # allow arbitrary extra attributes for extensibility
    extras: Any = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}

    def get(self, name: str, default=None):
        return getattr(self, name, self.extras.get(name, default))
