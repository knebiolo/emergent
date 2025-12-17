"""Behavior manager composes primitives into high-level commands."""
from typing import Dict, Any
import numpy as np

from . import primitives


class BehaviorManager:
    def __init__(self, weights: Dict[str, float] = None):
        # weights: e.g., {'schooling': 1.0, 'collision': 1.0}
        self.weights = weights or {'schooling': 1.0, 'collision': 1.0}

    def step(self, positions: np.ndarray, headings: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """Compute desired commands for a single timestep.

        Returns dict with keys: 'desired_heading_vec' (n,2), 'desired_speed' (n,)
        """
        # schooling
        school_vec = primitives.schooling_vector(positions, radii=0.0, ideal_dist=1.0)
        coll_vec = primitives.collision_avoidance_vector(positions, min_sep=0.5)

        combined = self.weights.get('schooling', 1.0) * school_vec + self.weights.get('collision', 1.0) * coll_vec

        # desired heading: angle of combined vector; desired speed = magnitude
        desired_heading_vec = combined
        desired_speed = np.linalg.norm(combined, axis=1)
        return {'desired_heading_vec': desired_heading_vec, 'desired_speed': desired_speed}
