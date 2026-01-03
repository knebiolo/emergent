"""
Reinforcement Learning for Behavioral Weight Optimization

This module implements RL training infrastructure for learning instinctual
behavioral parameters that produce realistic schooling and upstream migration.

Key architecture: Separate persistent instincts (learned once) from ephemeral
spatial state (reset each simulation).

Based on:
- docs/RL_TRAINING_GUIDE.md
- docs/BIOLOGICAL_SCHOOLING_METRICS.md
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy.spatial import cKDTree


@dataclass
class BehavioralWeights:
    """
    Persistent instinctual parameters learned through RL training.
    
    These weights control emergent behavior and are learned once, then reused
    across many simulations. Spatial state (positions, velocities, memory) is
    reset between training episodes.
    
    Biological basis:
    - Sensory range: 2 BL (Partridge & Pitcher 1980)
    - Cohesion distance: 2.0-3.0 BL relaxed, 1.5 BL threatened (Magurran & Pitcher 1987)
    - Drafting benefits: 15% single file, 25% V-formation (Weihs 1973, Fish & Lauder 2006)
    """
    
    # Schooling dynamics
    cohesion_weight: float = 1000.0
    alignment_weight: float = 25000.0
    separation_weight: float = 5000.0
    separation_radius: float = 1.0  # Body lengths
    
    # Environmental responses
    rheotaxis_weight: float = 25000.0
    border_cue_weight: float = 200000.0
    border_threshold_multiplier: float = 2.0
    border_max_force: float = 10.0
    
    # Collision avoidance
    collision_weight: float = 2000.0
    collision_radius: float = 0.5  # Body lengths
    
    # Additional behavioral weights
    low_speed_weight: float = 1500.0
    wave_drag_weight: float = 0.0
    refugia_weight: float = 50000.0
    shallow_weight: float = 500000.0
    avoid_weight: float = 25000.0
    
    # Sensory and threat parameters
    sensory_range: float = 2.0  # Body lengths (biological constant)
    threat_level: float = 0.3  # 0.0 = relaxed, 1.0 = high threat
    
    # Dynamic cohesion parameters (threat-responsive)
    cohesion_radius_relaxed: float = 3.0  # Body lengths
    cohesion_radius_threatened: float = 1.5  # Body lengths
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BehavioralWeights':
        """Load from dictionary (JSON deserialization)."""
        return cls(**data)
    
    def to_json(self, filepath: Path) -> None:
        """Save weights to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'BehavioralWeights':
        """Load weights from JSON file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Behavioral weights file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_test_weights_dict(self) -> Dict[str, float]:
        """
        Convert to format used by simulation.test_weights.
        
        Maps BehavioralWeights attribute names to the keys expected by
        tools/run_nuyakuk_headless.py and simulation.py.
        """
        return {
            'rheotaxis': self.rheotaxis_weight,
            'alignment': self.alignment_weight,
            'cohesion': self.cohesion_weight,
            'collision': self.collision_weight,
            'low_speed': self.low_speed_weight,
            'wave_drag': self.wave_drag_weight,
            'refugia': self.refugia_weight,
            'border': self.border_cue_weight,
            'shallow': self.shallow_weight,
            'avoid': self.avoid_weight,
        }
    
    def validate(self) -> None:
        """
        Validate weights are within reasonable ranges.
        
        Raises:
            ValueError: If any weight is negative or outside expected range.
        """
        # Check non-negative weights
        for field_name, value in self.to_dict().items():
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        
        # Check specific constraints
        if self.sensory_range < 0.5 or self.sensory_range > 5.0:
            raise ValueError(f"sensory_range should be 0.5-5.0 BL (biological), got {self.sensory_range}")
        
        if self.threat_level < 0.0 or self.threat_level > 1.0:
            raise ValueError(f"threat_level must be 0.0-1.0, got {self.threat_level}")
        
        if self.cohesion_radius_relaxed < self.cohesion_radius_threatened:
            raise ValueError(
                f"cohesion_radius_relaxed ({self.cohesion_radius_relaxed}) must be >= "
                f"cohesion_radius_threatened ({self.cohesion_radius_threatened})"
            )
    
    def mutate(self, mutation_scale: float = 0.1, rng: Optional[np.random.Generator] = None) -> 'BehavioralWeights':
        """
        Create mutated copy for RL exploration.
        
        Applies Gaussian perturbation to all weights for training exploration.
        
        Args:
            mutation_scale: Standard deviation as fraction of current value (default 0.1 = 10%)
            rng: Random number generator (default: creates new one)
        
        Returns:
            New BehavioralWeights instance with mutated values.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        data = self.to_dict()
        mutated = {}
        
        for key, value in data.items():
            # Gaussian perturbation: N(value, mutation_scale * value)
            noise = rng.normal(0, mutation_scale * abs(value))
            mutated[key] = max(0.0, value + noise)  # Clip to non-negative
        
        return BehavioralWeights.from_dict(mutated)


# =============================================================================
# Schooling Quality Metrics
# =============================================================================

def compute_cohesion_score(
    positions: np.ndarray,
    body_length: float,
    threat_level: float = 0.3,
    sensory_range: float = 2.0
) -> np.ndarray:
    """
    Compute cohesion quality score for each agent.
    
    Measures proximity to ideal group spacing using local centroid of neighbors
    within sensory range (2 BL).
    
    Biological basis: Fish maintain threat-responsive spacing (Magurran & Pitcher 1987).
    - Relaxed: ~3.0 BL spacing
    - Threatened: ~1.5 BL spacing
    
    Args:
        positions: Agent positions, shape (N, 2) or (N, 3)
        body_length: Fish body length in meters
        threat_level: 0.0 = relaxed, 1.0 = high threat
        sensory_range: Neighbor detection range in body lengths (default 2.0)
    
    Returns:
        Cohesion scores, shape (N,). Range 0.0-1.0.
        - 1.0 = Perfect spacing at ideal distance
        - 0.5 = Moderate deviation (±0.5 BL)
        - 0.0 = Large deviation (>2 BL from ideal)
    """
    N = len(positions)
    if N == 0:
        return np.array([])
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(positions)
    search_radius = sensory_range * body_length
    
    # Ideal distance adjusted by threat level
    # Relaxed: 2.0 BL, Threatened: 1.0 BL (linear interpolation)
    ideal_dist = body_length * (2.0 - threat_level)
    
    cohesion_scores = np.zeros(N)
    
    for i in range(N):
        # Find neighbors within sensory range
        neighbor_indices = tree.query_ball_point(positions[i], r=search_radius)
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]
        
        if len(neighbor_indices) == 0:
            cohesion_scores[i] = 0.0  # Isolated agent
            continue
        
        # Compute local centroid
        neighbor_positions = positions[neighbor_indices]
        centroid = np.mean(neighbor_positions, axis=0)
        
        # Distance to centroid
        dist_to_centroid = np.linalg.norm(positions[i] - centroid)
        
        # Gaussian reward centered at ideal distance
        # σ = 0.5 BL (controls width of reward peak)
        sigma = 0.5 * body_length
        cohesion_scores[i] = np.exp(-0.5 * ((dist_to_centroid - ideal_dist) / sigma)**2)
    
    return cohesion_scores


def compute_alignment_score(
    headings: np.ndarray,
    positions: np.ndarray,
    body_length: float,
    sensory_range: float = 2.0
) -> np.ndarray:
    """
    Compute heading alignment score for each agent.
    
    Measures directional coordination with neighbors using circular mean.
    
    Args:
        headings: Agent headings in radians, shape (N,)
        positions: Agent positions, shape (N, 2) or (N, 3)
        body_length: Fish body length in meters
        sensory_range: Neighbor detection range in body lengths (default 2.0)
    
    Returns:
        Alignment scores, shape (N,). Range -1.0 to 1.0.
        - 1.0 = Perfect alignment (same direction as neighbors)
        - 0.0 = Perpendicular (90° difference)
        - -1.0 = Opposite direction (180° difference)
    """
    N = len(positions)
    if N == 0:
        return np.array([])
    
    # Build KD-tree for neighbor search
    tree = cKDTree(positions)
    search_radius = sensory_range * body_length
    
    alignment_scores = np.zeros(N)
    
    for i in range(N):
        # Find neighbors within sensory range
        neighbor_indices = tree.query_ball_point(positions[i], r=search_radius)
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]
        
        if len(neighbor_indices) == 0:
            alignment_scores[i] = 0.0  # Isolated agent
            continue
        
        # Circular mean of neighbor headings
        neighbor_headings = headings[neighbor_indices]
        mean_heading = np.arctan2(
            np.mean(np.sin(neighbor_headings)),
            np.mean(np.cos(neighbor_headings))
        )
        
        # Angular difference to my heading
        heading_diff = headings[i] - mean_heading
        
        # Cosine similarity (wraps correctly for angles)
        alignment_scores[i] = np.cos(heading_diff)
    
    return alignment_scores


def compute_separation_penalty(
    positions: np.ndarray,
    body_length: float
) -> np.ndarray:
    """
    Compute separation penalty for agents too close together.
    
    Penalizes crowding when agents are closer than 1.0 BL.
    
    Args:
        positions: Agent positions, shape (N, 2) or (N, 3)
        body_length: Fish body length in meters
    
    Returns:
        Separation penalties, shape (N,). Range -1.0 to 0.0.
        - 0.0 = No crowding (>1 BL clearance)
        - -0.5 = Moderate crowding (0.5 BL apart)
        - -1.0 = Severe crowding (touching)
    """
    N = len(positions)
    if N == 0:
        return np.array([])
    
    # Build KD-tree
    tree = cKDTree(positions)
    
    # Query 2 nearest neighbors (self + closest other)
    distances, _ = tree.query(positions, k=2)
    
    # distances[:, 1] is distance to nearest neighbor (not self)
    nearest_neighbor_dist = distances[:, 1]
    
    # Penalty if closer than 1.0 BL
    crowding_threshold = 1.0 * body_length
    penalties = np.where(
        nearest_neighbor_dist < crowding_threshold,
        -(crowding_threshold - nearest_neighbor_dist) / body_length,  # Linear penalty
        0.0
    )
    
    return penalties


def compute_overall_schooling_score(
    positions: np.ndarray,
    headings: np.ndarray,
    body_length: float,
    threat_level: float = 0.3,
    sensory_range: float = 2.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute overall schooling quality combining cohesion, alignment, separation.
    
    Args:
        positions: Agent positions, shape (N, 2) or (N, 3)
        headings: Agent headings in radians, shape (N,)
        body_length: Fish body length in meters
        threat_level: 0.0 = relaxed, 1.0 = high threat
        sensory_range: Neighbor detection range in body lengths (default 2.0)
    
    Returns:
        Tuple of (overall_score, component_dict):
        - overall_score: Mean across all agents. Range -1 to 2.
          - Excellent: ~2.0 (perfect cohesion + alignment, no crowding)
          - Good: 1.0-1.5
          - Poor: <0.5
          - Dysfunctional: <0.0
        - component_dict: Individual metric means for debugging
    """
    cohesion = compute_cohesion_score(positions, body_length, threat_level, sensory_range)
    alignment = compute_alignment_score(headings, positions, body_length, sensory_range)
    separation = compute_separation_penalty(positions, body_length)
    
    # Overall score per agent
    per_agent_scores = cohesion + alignment + separation
    
    # Mean across population
    overall_mean = np.mean(per_agent_scores)
    
    components = {
        'cohesion': np.mean(cohesion),
        'alignment': np.mean(alignment),
        'separation': np.mean(separation),
        'overall': overall_mean
    }
    
    return overall_mean, components
