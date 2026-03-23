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
from typing import Dict, Optional, Tuple, Callable, Any
from pathlib import Path
from scipy.spatial import cKDTree
import time


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
    
    # Environmental responses
    rheotaxis_weight: float = 25000.0
    border_cue_weight: float = 200000.0
    
    # Collision avoidance
    collision_weight: float = 2000.0
    separation_weight: float = 1500.0
    
    # Additional behavioral weights
    low_speed_weight: float = 1500.0
    wave_drag_weight: float = 0.0
    refugia_weight: float = 50000.0
    shallow_weight: float = 500000.0
    avoid_weight: float = 25000.0
    
    # Arbitration parameters
    arbitration_tolerance: float = 50000.0  # Tolerance for cue accumulation (how many 'f4cks' a fish has)
    
    # Threat parameters (FIXED - not trainable)
    threat_level: float = 0.0  # Default relaxed; set to 1.0 for tight schooling
    
    # Dynamic cohesion parameters (threat-responsive)
    cohesion_radius_relaxed: float = 3.0  # Body lengths
    cohesion_radius_threatened: float = 1.5  # Body lengths
    
    # Jump/leap behavior (for fish in high-velocity regions)
    jump_velocity_ratio_threshold: float = 0.10  # Jump when SOG/water_velocity < 10%
    jump_battery_threshold: float = 0.25  # Minimum battery level to jump (25%)
    jump_angle_min_deg: float = 45.0  # Minimum jump angle in degrees
    jump_angle_max_deg: float = 60.0  # Maximum jump angle in degrees
    
    # Cue application order (indices 0-9 map to cue names)
    # Default order: shallow, border, avoid, collision, alignment, cohesion, low_speed, refugia, rheotaxis, wave_drag
    order_0: int = 0  # First cue to apply
    order_1: int = 1
    order_2: int = 2
    order_3: int = 3
    order_4: int = 4
    order_5: int = 5
    order_6: int = 6
    order_7: int = 7
    order_8: int = 8
    order_9: int = 9  # Last cue to apply
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralWeights':
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
        simulation.test_weights (legacy/deprecated runners may also rely on these keys).
        """
        return {
            # Behavioral cue weights
            'rheotaxis': self.rheotaxis_weight,
            'alignment': self.alignment_weight,
            'cohesion': self.cohesion_weight,
            'separation': self.separation_weight,
            'collision': self.collision_weight,
            'low_speed': self.low_speed_weight,
            'wave_drag': self.wave_drag_weight,
            'refugia': self.refugia_weight,
            'border': self.border_cue_weight,
            'shallow': self.shallow_weight,
            'avoid': self.avoid_weight,
            
            # Threat-responsive schooling parameters (ACTUALLY IMPLEMENTED)
            'threat_level': self.threat_level,
            'cohesion_radius_relaxed': self.cohesion_radius_relaxed,
            'cohesion_radius_threatened': self.cohesion_radius_threatened,
            
            # Jump/leap parameters
            'jump_velocity_ratio_threshold': self.jump_velocity_ratio_threshold,
            'jump_battery_threshold': self.jump_battery_threshold,
            'jump_angle_min_deg': self.jump_angle_min_deg,
            'jump_angle_max_deg': self.jump_angle_max_deg,
            
            # Cue application order
            'order_0': self.order_0,
            'order_1': self.order_1,
            'order_2': self.order_2,
            'order_3': self.order_3,
            'order_4': self.order_4,
            'order_5': self.order_5,
            'order_6': self.order_6,
            'order_7': self.order_7,
            'order_8': self.order_8,
            'order_9': self.order_9,
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
        if self.threat_level < 0.0 or self.threat_level > 1.0:
            raise ValueError(f"threat_level must be 0.0-1.0, got {self.threat_level}")
        
        if self.cohesion_radius_relaxed < self.cohesion_radius_threatened:
            raise ValueError(
                f"cohesion_radius_relaxed ({self.cohesion_radius_relaxed}) must be >= "
                f"cohesion_radius_threatened ({self.cohesion_radius_threatened})"
            )
        
        # Jump parameter validation
        if self.jump_velocity_ratio_threshold < 0.0 or self.jump_velocity_ratio_threshold > 1.0:
            raise ValueError(f"jump_velocity_ratio_threshold must be 0.0-1.0, got {self.jump_velocity_ratio_threshold}")
        
        if self.jump_battery_threshold < 0.0 or self.jump_battery_threshold > 1.0:
            raise ValueError(f"jump_battery_threshold must be 0.0-1.0, got {self.jump_battery_threshold}")
        
        if self.jump_angle_min_deg < 0.0 or self.jump_angle_min_deg > 90.0:
            raise ValueError(f"jump_angle_min_deg must be 0-90 degrees, got {self.jump_angle_min_deg}")
        
        if self.jump_angle_max_deg < 0.0 or self.jump_angle_max_deg > 90.0:
            raise ValueError(f"jump_angle_max_deg must be 0-90 degrees, got {self.jump_angle_max_deg}")
        
        if self.jump_angle_min_deg > self.jump_angle_max_deg:
            raise ValueError(
                f"jump_angle_min_deg ({self.jump_angle_min_deg}) must be <= "
                f"jump_angle_max_deg ({self.jump_angle_max_deg})"
            )
    
    def randomize(self, scale: float = 0.5, rng: Optional[np.random.Generator] = None) -> 'BehavioralWeights':
        """
        Create randomized copy for initial exploration.
        
        Uses larger perturbations than mutate() for initial diversity.
        
        Args:
            scale: Randomization scale (default 0.5 = 50% variation)
            rng: Random number generator (default: creates new one)
        
        Returns:
            New BehavioralWeights instance with randomized values.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        data = self.to_dict()
        randomized = {}
        
        # Fixed parameters (not trainable)
        fixed_params = {'threat_level'}
        
        for key, value in data.items():
            if key in fixed_params:
                # Keep fixed parameters unchanged
                randomized[key] = value
            elif isinstance(value, bool):
                randomized[key] = value
            elif isinstance(value, (int, float)):
                # Apply larger random perturbation for initial exploration
                perturbation = rng.normal(0, abs(value) * scale)
                new_value = value + perturbation
                # Ensure non-zero: clip to minimum 10.0 for weights (visible changes)
                randomized[key] = max(10.0, new_value) if new_value >= 0 else 10.0
            else:
                randomized[key] = value
        
        return BehavioralWeights.from_dict(randomized)
    
    def mutate(self, mutation_scale: float = 0.1, order_mutation_prob: float = 0.2, rng: Optional[np.random.Generator] = None) -> 'BehavioralWeights':
        """
        Create mutated copy for RL exploration.
        
        Applies Gaussian perturbation to weights and occasionally swaps cue application order.
        
        Args:
            mutation_scale: Standard deviation as fraction of current value (default 0.1 = 10%)
            order_mutation_prob: Probability of mutating cue order (default 0.2 = 20% chance)
            rng: Random number generator (default: creates new one)
        
        Returns:
            New BehavioralWeights instance with mutated values.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        data = self.to_dict()
        mutated = {}
        
        # Extract current order as array
        current_order = np.array([
            data.get('order_0', 0), data.get('order_1', 1), data.get('order_2', 2),
            data.get('order_3', 3), data.get('order_4', 4), data.get('order_5', 5),
            data.get('order_6', 6), data.get('order_7', 7), data.get('order_8', 8),
            data.get('order_9', 9)
        ], dtype=int)
        
        # Mutate order with some probability (swap-based mutation)
        if rng.random() < order_mutation_prob:
            # Strategy: swap 2-3 random positions to explore order changes
            num_swaps = rng.integers(1, 4)  # 1-3 swaps
            for _ in range(num_swaps):
                i, j = rng.choice(10, size=2, replace=False)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Store mutated order
        for idx in range(10):
            mutated[f'order_{idx}'] = int(current_order[idx])
        
        # Fixed parameters (not trainable) - user-configured thresholds
        fixed_params = {'threat_level', 'arbitration_tolerance'}
        
        # Mutate all other fields (weights, thresholds, etc.)
        for key, value in data.items():
            if key in fixed_params:
                # Keep fixed parameters unchanged
                mutated[key] = value
            elif not key.startswith('order_'):
                # Gaussian perturbation: N(value, mutation_scale * value)
                noise = rng.normal(0, mutation_scale * abs(value))
                new_value = value + noise
                # Ensure non-zero: clip to minimum 10.0 for weights (visible changes)
                mutated[key] = max(10.0, new_value) if new_value >= 0 else 10.0
        
        return BehavioralWeights.from_dict(mutated)


# =============================================================================
# Schooling Quality Metrics
# =============================================================================

def compute_cohesion_score(
    positions: np.ndarray,
    body_length: float,
    threat_level: float = 0.3,
    sensory_range: float = 5.0
) -> np.ndarray:
    """
    Compute cohesion quality score for each agent.
    
    Measures proximity to ideal group spacing using local centroid of neighbors
    within sensory range (5 BL = 1.5m).
    
    Biological basis: Fish maintain threat-responsive spacing (Magurran & Pitcher 1987).
    - Relaxed: ~3.0 BL spacing
    - Threatened: ~1.5 BL spacing
    
    Args:
        positions: Agent positions, shape (N, 2) or (N, 3)
        body_length: Fish body length in meters
        threat_level: 0.0 = relaxed, 1.0 = high threat
        sensory_range: Neighbor detection range in body lengths (default 5.0)
    
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
    
    # DEBUG: Track statistics for diagnostics (very low sampling rate)
    debug_sample = N > 0 and np.random.random() < 0.0001  # 0.01% sampling
    neighbor_counts = []
    distances = []
    scores = []
    
    for i in range(N):
        # Find neighbors within sensory range
        neighbor_indices = tree.query_ball_point(positions[i], r=search_radius)
        neighbor_indices = [idx for idx in neighbor_indices if idx != i]
        
        if debug_sample:
            neighbor_counts.append(len(neighbor_indices))
        
        if len(neighbor_indices) == 0:
            cohesion_scores[i] = 0.0  # Isolated agent
            continue
        
        # Compute local centroid
        neighbor_positions = positions[neighbor_indices]
        centroid = np.mean(neighbor_positions, axis=0)
        
        # Distance to centroid
        dist_to_centroid = np.linalg.norm(positions[i] - centroid)
        
        # Gaussian reward centered at ideal distance
        # σ = 1.0 BL (controls width of reward peak)
        # Wider sigma allows more tolerance for spacing variation
        sigma = 1.0 * body_length
        cohesion_scores[i] = np.exp(-0.5 * ((dist_to_centroid - ideal_dist) / sigma)**2)
        
        if debug_sample:
            distances.append(dist_to_centroid)
            scores.append(cohesion_scores[i])
    
    if debug_sample and neighbor_counts:
        print(f"[COHESION DEBUG] N={N}, neighbors={np.mean(neighbor_counts):.1f}±{np.std(neighbor_counts):.1f}, "
              f"dist={np.mean(distances):.2f}±{np.std(distances):.2f}m (ideal={ideal_dist:.2f}m), "
              f"score={np.mean(scores):.3f}±{np.std(scores):.3f}, sum={np.sum(cohesion_scores):.1f}")
    
    return cohesion_scores


def compute_alignment_score(
    headings: np.ndarray,
    positions: np.ndarray,
    body_length: float,
    sensory_range: float = 5.0
) -> np.ndarray:
    """
    Compute heading alignment score for each agent.
    
    Measures directional coordination with neighbors using circular mean.
    
    Args:
        headings: Agent headings in radians, shape (N,)
        positions: Agent positions, shape (N, 2) or (N, 3)
        body_length: Fish body length in meters
        sensory_range: Neighbor detection range in body lengths (default 5.0)
    
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


# =============================================================================
# Episode Reward Function
# =============================================================================

def compute_episode_reward(
    positions_history: np.ndarray,
    headings_history: np.ndarray,
    velocities_history: np.ndarray,
    alive_history: np.ndarray,
    body_length: float,
    threat_level: float = 0.3,
    boundary_coords: Optional[np.ndarray] = None,
    boundary_threshold: float = 2.0,
    behavioral_weights: Optional[Dict[str, float]] = None,
    battery_history: Optional[np.ndarray] = None,
    longitudinal_profile: Optional[Any] = None,
    velocity_field_history: Optional[np.ndarray] = None,
    reward_weights: Optional[Dict[str, float]] = None,
    phase_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total reward for a training episode.
    
    Combines schooling quality, upstream progress, energy efficiency, and penalties
    for boundary violations and mortality.
    
    Based on BIOLOGICAL_SCHOOLING_METRICS.md reward function (lines 179-213).
    
    Args:
        positions_history: Agent positions over time, shape (T, N, 2) or (T, N, 3)
        headings_history: Agent headings over time, shape (T, N)
        velocities_history: Agent velocities over time, shape (T, N, 2) or (T, N, 3)
        alive_history: Agent alive status over time, shape (T, N) boolean
        body_length: Fish body length in meters
        threat_level: 0.0 = relaxed, 1.0 = high threat
        boundary_coords: Optional boundary polygon vertices, shape (M, 2)
        boundary_threshold: Distance to boundary considered "near" (meters)
        phase_callback: Optional callback for coarse progress updates during scoring
    
    Returns:
        Tuple of (total_reward, components_dict):
        - total_reward: Scalar reward for episode
          - Perfect: 20-30 (tight formation, efficient migration)
          - Good: 10-20 (coordinated but imperfect)
          - Poor: 0-10 (fragmented, inefficient)
          - Catastrophic: <0 (high mortality, collisions)
        - components_dict: Breakdown of reward components
    
    Component weights (from BIOLOGICAL_SCHOOLING_METRICS.md):
    - Cohesion: 10.0 (core schooling quality)
    - Alignment: 5.0 (heading coordination, shifted -1:1 → 0:2)
    - Separation: 5.0 (crowding penalty)
    - Upstream progress: 0.5 (meters/step)
    - Energy efficiency: 2.0 (m/kcal)
    - Drafting benefit: 20.0 (formation quality, 0.0-0.25 range)
    - Boundary proximity: -5.0 per agent (safety penalty)
    - Mortality: -50.0 per death (strong survival penalty)
    - Smoothness: -0.2 (Δaccel penalty for jerky movement)
    """
    # Use provided reward weights or defaults
    if reward_weights is None:
        reward_weights = {
            'cohesion': 0.1,
            'alignment': 1.0,
            'separation': -0.2,
            'upstream_progress': 10.0,
            'energy_efficiency': 2.0,
            'drafting_benefit': 20.0,
            'boundary_penalty': -10.0,
            'mortality_penalty': -50.0,
            'smoothness_penalty': -0.2,
            'fatigue_penalty': -0.9,
            'stagnation_penalty': -0.9,
            'rheotaxis_alignment': 10.0,
        }
    
    T, N = positions_history.shape[0], positions_history.shape[1]
    
    if T == 0 or N == 0:
        return 0.0, {}

    def _emit_phase(message: str) -> None:
        if phase_callback is None:
            return
        try:
            phase_callback(str(message))
        except Exception:
            # Best-effort status hook; never fail reward computation on callback issues.
            pass

    def _emit_progress(prefix: str, idx: int, total: int) -> None:
        if phase_callback is None or total <= 0:
            return
        step = idx + 1
        interval = max(1, total // 4)
        if step == 1 or step == total or (step % interval == 0):
            _emit_phase(f"{prefix} {step}/{total}")
    
    # =================================================================
    # 1. Schooling Quality (averaged over time)
    # =================================================================
    _emit_phase("scoring: schooling metrics")
    cohesion_scores = []
    alignment_scores = []
    separation_penalties = []
    
    for t in range(T):
        _emit_progress("scoring: schooling", t, T)
        positions_t = positions_history[t]
        headings_t = headings_history[t]
        alive_t = alive_history[t]
        
        # Only compute for alive agents
        if np.sum(alive_t) > 0:
            pos_alive = positions_t[alive_t]
            head_alive = headings_t[alive_t]
            
            cohesion = compute_cohesion_score(pos_alive, body_length, threat_level)
            alignment = compute_alignment_score(head_alive, pos_alive, body_length)
            separation = compute_separation_penalty(pos_alive, body_length)
            
            # Sum across agents (we want total school performance, not per-agent)
            cohesion_scores.append(np.sum(cohesion))
            alignment_scores.append(np.sum(alignment))
            separation_penalties.append(np.sum(separation))
    
    # Total sum across all agent-timesteps
    sum_cohesion = np.sum(cohesion_scores) if cohesion_scores else 0.0
    sum_alignment = np.sum(alignment_scores) if alignment_scores else 0.0
    sum_separation = np.sum(separation_penalties) if separation_penalties else 0.0
    
    # DEBUG: Print schooling metrics summary (rare samples only)
    if np.random.random() < 0.001:  # 0.1% sampling rate
        print(f"[SCHOOLING METRICS] T={T}, N={N}, "
              f"cohesion_sum={sum_cohesion:.1f} (×0.01={sum_cohesion*0.01:.2f}), "
              f"alignment_sum={sum_alignment:.1f} (×0.01={sum_alignment*0.01:.2f}), "
              f"separation_sum={sum_separation:.1f}")
    
    # =================================================================
    # 2. Upstream Progress (Flow Vector Integration)
    # =================================================================
    _emit_phase("scoring: upstream progress")
    # Use flow vector integration: measures distance swum AGAINST current
    # Works for braided channels, pools, and any river geometry
    # Formula: upstream_progress = sum(displacement · upstream_unit_vector)
    # where upstream_unit_vector = -velocity / |velocity|
    
    if velocity_field_history is None:
        raise ValueError(
            "velocity_field_history is required for computing upstream progress. "
            "Flow vector integration replaces longitudinal profile projection to handle "
            "braided channels and complex geometry. Pass sampled velocity at agent positions."
        )
    
    # Compute upstream progress via flow integration
    total_upstream_progress = 0.0
    
    for t in range(1, T):
        _emit_progress("scoring: upstream", t - 1, T - 1)
        alive_t = alive_history[t]
        if np.sum(alive_t) == 0:
            continue
        
        # Get alive agents at this timestep
        positions_t = positions_history[t, alive_t]
        positions_prev = positions_history[t-1, alive_t]
        velocities_t = velocity_field_history[t, alive_t]  # Sampled water velocity
        
        # Displacement of each agent
        displacement = positions_t - positions_prev  # Shape: (n_alive, 2)
        
        # Water velocity magnitude at each agent
        vel_mag = np.linalg.norm(velocities_t, axis=1)  # Shape: (n_alive,)
        
        # Upstream unit vector = -velocity / |velocity|
        # Only compute where velocity is significant (>0.01 m/s)
        valid = vel_mag > 0.01
        
        if np.any(valid):
            upstream_unit = np.zeros_like(velocities_t)
            upstream_unit[valid] = -velocities_t[valid] / vel_mag[valid, np.newaxis]
            
            # Project displacement onto upstream direction (dot product)
            # progress > 0 = moved against current (upstream)
            # progress < 0 = moved with current (downstream/drift)
            # progress = 0 = moved perpendicular to flow
            progress_per_agent = np.sum(displacement * upstream_unit, axis=1)
            
            # Sum total progress across all agents this timestep
            total_upstream_progress += np.sum(progress_per_agent[valid])
    
    # Total upstream distance summed across all agents over entire episode (meters)
    # This gives proper magnitude: 100 fish × 100 timesteps × 0.05m = 500m total
    sum_upstream_progress = total_upstream_progress
    
    # =================================================================
    # 3. Energy Efficiency (distance / speed²)
    # =================================================================
    _emit_phase("scoring: energy efficiency")
    # Energy ∝ speed², so efficiency = distance traveled / sum(speed²)
    total_distance = 0.0
    total_energy = 0.0
    
    for t in range(1, T):
        _emit_progress("scoring: energy", t - 1, T - 1)
        alive_t = alive_history[t]
        if np.sum(alive_t) == 0:
            continue
        
        # Distance traveled by alive agents
        displacement = positions_history[t, alive_t] - positions_history[t-1, alive_t]
        distances = np.linalg.norm(displacement, axis=1)
        total_distance += np.sum(distances)
        
        # Energy (proportional to speed²)
        speeds = np.linalg.norm(velocities_history[t, alive_t], axis=1)
        total_energy += np.sum(speeds ** 2)
    
    energy_efficiency = total_distance / total_energy if total_energy > 0 else 0.0
    
    # =================================================================
    # 4. Drafting Benefit (placeholder - would need formation detection)
    # =================================================================
    # TODO: Implement drafting detection (agents swimming in formation)
    # For now, assume no drafting benefit
    mean_drafting_benefit = 0.0
    
    # =================================================================
    # 5. Boundary Proximity Penalty
    # =================================================================
    _emit_phase("scoring: boundary proximity")
    agents_near_boundary = 0
    if boundary_coords is not None:
        # Count timesteps where agents are near boundary
        for t in range(T):
            _emit_progress("scoring: boundary", t, T)
            alive_t = alive_history[t]
            if np.sum(alive_t) == 0:
                continue
            
            pos_alive = positions_history[t, alive_t]
            
            # Simplified: check distance to any boundary point
            # (In practice, would use point-to-polygon distance)
            for pos in pos_alive:
                min_dist = np.min(np.linalg.norm(boundary_coords - pos, axis=1))
                if min_dist < boundary_threshold:
                    agents_near_boundary += 1
    
    # =================================================================
    # 6. Mortality Penalty
    # =================================================================
    _emit_phase("scoring: mortality")
    initial_alive = np.sum(alive_history[0])
    final_alive = np.sum(alive_history[-1])
    dead_count = initial_alive - final_alive
    
    # =================================================================
    # 7. Movement Smoothness (acceleration changes)
    # =================================================================
    _emit_phase("scoring: smoothness")
    # Compute acceleration changes (jerk)
    accel_smoothness_penalty = 0.0
    
    for t in range(2, T):
        _emit_progress("scoring: smoothness", t - 2, T - 2)
        alive_t = alive_history[t]
        if np.sum(alive_t) == 0:
            continue
        
        # Velocity changes (acceleration)
        vel_curr = velocities_history[t, alive_t]
        vel_prev = velocities_history[t-1, alive_t]
        vel_prev2 = velocities_history[t-2, alive_t]
        
        accel_curr = vel_curr - vel_prev
        accel_prev = vel_prev - vel_prev2
        
        # Jerk = change in acceleration
        jerk = np.linalg.norm(accel_curr - accel_prev, axis=1)
        accel_smoothness_penalty += np.sum(jerk)
    
    # Total jerk summed across all agents and timesteps
    
    # =================================================================
    # 8. Fatigue Penalty (CRITICAL for preventing exhaustion)
    # =================================================================
    _emit_phase("scoring: fatigue")
    # Heavily penalize low battery states to incentivize energy management
    fatigue_penalty = 0.0
    if battery_history is not None:
        # Battery is 0.0 (depleted) to 1.0 (full)
        # Penalize time spent below threshold
        low_battery_threshold = 0.3  # Below 30% is critical
        for t in range(T):
            _emit_progress("scoring: fatigue", t, T)
            alive_t = alive_history[t]
            if np.sum(alive_t) == 0:
                continue
            
            battery_t = battery_history[t, alive_t]
            # Count agent-timesteps below threshold
            low_battery_count = np.sum(battery_t < low_battery_threshold)
            fatigue_penalty += low_battery_count
            
            # Extra penalty for completely depleted (battery = 0)
            depleted_count = np.sum(battery_t <= 0.01)
            fatigue_penalty += depleted_count * 2.0  # Double penalty for full depletion
        
        # Total fatigue penalty summed across all agent-timesteps
    
    # =================================================================
    # 8. Rheotaxis Alignment (Swimming into Flow)
    # =================================================================
    _emit_phase("scoring: stagnation")
    # Penalize stationary fish (vibrating in place / no movement)
    # Fish should actively swim, not form static lattices
    # Compute average displacement per timestep
    
    stagnation_penalty = 0.0
    
    for t in range(1, T):
        _emit_progress("scoring: stagnation", t - 1, T - 1)
        alive_t = alive_history[t]
        alive_prev = alive_history[t-1]
        alive_both = alive_t & alive_prev
        
        if np.sum(alive_both) > 0:
            positions_t = positions_history[t, alive_both]
            positions_prev = positions_history[t-1, alive_both]
            
            # Compute displacement magnitude
            displacement = np.linalg.norm(positions_t - positions_prev, axis=1)
            
            # Penalize very small displacements (< 0.1 m per timestep)
            # Fish should move at least ~0.1-0.5 m/s minimum
            stagnant_mask = displacement < 0.1  # Less than 10cm movement
            stagnation_penalty += np.sum(stagnant_mask)
    
    # Total stagnation count summed across all agent-timesteps
    
    # =================================================================
    # Soft rheotaxis orientation bonus.
    # Fish should generally orient against flow, but we intentionally avoid a
    # strict per-step penalty so other cues can still steer local maneuvers.
    
    _emit_phase("scoring: rheotaxis alignment")
    rheotaxis_alignment_bonus = 0.0
    rheotaxis_alignment_sum = 0.0
    rheotaxis_alignment_count = 0
    
    if velocity_field_history is not None:
        for t in range(T):
            _emit_progress("scoring: rheotaxis", t, T)
            alive_t = alive_history[t]
            if np.sum(alive_t) == 0:
                continue
            
            headings_t = headings_history[t, alive_t]
            velocities_t = velocity_field_history[t, alive_t]
            
            # Compute water velocity magnitude
            vel_mag = np.linalg.norm(velocities_t, axis=1)
            
            # Only penalize where flow is significant (>0.1 m/s)
            # In still water, fish can swim any direction
            significant_flow = vel_mag > 0.1
            
            if np.any(significant_flow):
                # Compute upstream angle (opposite of flow)
                flow_angle = np.arctan2(velocities_t[:, 1], velocities_t[:, 0])
                upstream_angle = flow_angle + np.pi  # Opposite direction
                
                # cos(angle_diff) = 1 upstream-aligned, 0 perpendicular, -1 downstream.
                angle_diff = headings_t - upstream_angle
                alignment = np.cos(angle_diff)
                
                # Soft bonus: reward only upstream-leaning alignment.
                # Perpendicular/downstream headings get no bonus (but no hard penalty).
                alignment_bonus = np.maximum(alignment[significant_flow], 0.0)
                rheotaxis_alignment_sum += float(np.sum(alignment_bonus))
                rheotaxis_alignment_count += int(alignment_bonus.size)
        
        # Normalize so scale does not explode with num_agents * timesteps.
        if rheotaxis_alignment_count > 0:
            rheotaxis_alignment_bonus = rheotaxis_alignment_sum / float(rheotaxis_alignment_count)
    
    # =================================================================
    # Total Reward Calculation
    # =================================================================
    _emit_phase("scoring: aggregation")
    
    # CRITICAL: Penalize zero/near-zero schooling weights to prevent degenerate solutions
    # where fish ignore each other and just follow rheotaxis in a line
    weight_diversity_bonus = 0.0
    min_schooling_weight_penalty = 0.0
    
    if behavioral_weights is not None:
        # Extract schooling-critical weights
        cohesion_w = behavioral_weights.get('cohesion', 0.0)
        alignment_w = behavioral_weights.get('alignment', 0.0)
        collision_w = behavioral_weights.get('collision', 0.0)
        
        # HARSH penalty if ANY schooling weight drops below threshold
        # Fish MUST school - it's a biological imperative!
        min_threshold = 1000.0  # Minimum acceptable weight
        if cohesion_w < min_threshold:
            min_schooling_weight_penalty -= 50.0  # Major penalty
        if alignment_w < min_threshold:
            min_schooling_weight_penalty -= 50.0
        if collision_w < min_threshold:
            min_schooling_weight_penalty -= 50.0
        
        # Bonus for weight diversity (prevents converging to single-cue solutions)
        all_weights = [
            behavioral_weights.get('rheotaxis', 0.0),
            cohesion_w,
            alignment_w,
            behavioral_weights.get('refugia', 0.0),
            collision_w,
            behavioral_weights.get('shallow', 0.0),
            behavioral_weights.get('low_speed', 0.0),
            behavioral_weights.get('wave_drag', 0.0),
            behavioral_weights.get('border', 0.0),
            behavioral_weights.get('avoid', 0.0),
        ]
        # Normalize to prevent scale bias
        total = sum(all_weights)
        if total > 0:
            normalized = [w / total for w in all_weights]
            # Entropy bonus: high entropy = diverse weights (good)
            # Low entropy = single dominant weight (bad)
            entropy = -sum(p * np.log(p + 1e-10) for p in normalized if p > 0)
            weight_diversity_bonus = entropy * 5.0  # Scale to ~5-10 range
    
    reward = (
        sum_cohesion * reward_weights['cohesion'] +
        sum_alignment * reward_weights['alignment'] +
        sum_separation * reward_weights['separation'] +
        sum_upstream_progress * reward_weights['upstream_progress'] +
        energy_efficiency * reward_weights['energy_efficiency'] +
        mean_drafting_benefit * reward_weights['drafting_benefit'] +
        agents_near_boundary * reward_weights['boundary_penalty'] +
        dead_count * reward_weights['mortality_penalty'] +
        accel_smoothness_penalty * reward_weights['smoothness_penalty'] +
        fatigue_penalty * reward_weights['fatigue_penalty'] +
        stagnation_penalty * reward_weights['stagnation_penalty'] +
        rheotaxis_alignment_bonus * reward_weights['rheotaxis_alignment'] +
        min_schooling_weight_penalty +
        weight_diversity_bonus
    )
    
    components = {
        'cohesion': sum_cohesion * reward_weights['cohesion'],
        'alignment': sum_alignment * reward_weights['alignment'],
        'separation': sum_separation * reward_weights['separation'],
        'upstream_progress': sum_upstream_progress * reward_weights['upstream_progress'],
        'energy_efficiency': energy_efficiency * reward_weights['energy_efficiency'],
        'drafting_benefit': mean_drafting_benefit * reward_weights['drafting_benefit'],
        'boundary_penalty': agents_near_boundary * reward_weights['boundary_penalty'],
        'mortality_penalty': dead_count * reward_weights['mortality_penalty'],
        'smoothness_penalty': accel_smoothness_penalty * reward_weights['smoothness_penalty'],
        'fatigue_penalty': fatigue_penalty * reward_weights['fatigue_penalty'],
        'stagnation_penalty': stagnation_penalty * reward_weights['stagnation_penalty'],
        'rheotaxis_alignment': rheotaxis_alignment_bonus * reward_weights['rheotaxis_alignment'],
        'min_schooling_penalty': min_schooling_weight_penalty,
        'weight_diversity_bonus': weight_diversity_bonus,
        'total': reward
    }
    _emit_phase("scoring: complete")
    
    return reward, components


# =============================================================================
# RL Training Infrastructure
# =============================================================================

class RLTrainer:
    """
    Reinforcement learning trainer for behavioral weight optimization.
    
    Uses evolutionary strategy with Gaussian perturbations to explore weight space
    and maximize episode rewards.
    
    Training workflow:
    1. Initialize with default behavioral weights
    2. Create simulation with current weights
    3. Run episode, collect trajectory
    4. Compute reward from episode history
    5. Mutate weights for exploration
    6. Keep best weights across all episodes
    7. Repeat for N episodes
    8. Save best weights to JSON
    """
    
    def __init__(
        self,
        simulation_factory: Callable[[BehavioralWeights], Any],
        initial_weights: Optional[BehavioralWeights] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize RL trainer.
        
        Args:
            simulation_factory: Function that takes BehavioralWeights and returns simulation instance
            initial_weights: Starting behavioral weights (default: BehavioralWeights())
            config: Training configuration dict with keys:
                - exploration_noise: Mutation stddev (default 0.1)
                - body_length: Fish body length in meters (default 0.5)
                - dt: Timestep duration in seconds (default 1.0)
                - num_timesteps: Steps per episode (optional, for factory)
                - reward_weights: Dict of reward multipliers (optional)
                - initial_heading_mode: 'environment', 'upstream', 'uniform', or 'downstream'
                - initial_sog_mode: 'environment', 'uniform', or 'fixed'
                - initial_sog_min: Minimum SOG for uniform/fixed modes (m/s)
                - initial_sog_max: Maximum SOG for uniform mode (m/s)
        """
        self.simulation_factory = simulation_factory
        self.initial_weights = initial_weights if initial_weights else BehavioralWeights()
        
        # Extract config
        config = config or {}
        self.exploration_noise = config.get('exploration_noise', 0.1)
        self.body_length = config.get('body_length', 0.5)
        self.dt = config.get('dt', 1.0)
        self.num_timesteps = config.get('num_timesteps', 100)
        self.initial_heading_mode = str(config.get('initial_heading_mode', 'environment')).strip().lower()
        self.initial_sog_mode = str(config.get('initial_sog_mode', 'environment')).strip().lower()
        self.initial_sog_min = float(config.get('initial_sog_min', 0.1))
        self.initial_sog_max = float(config.get('initial_sog_max', 1.5))
        if self.initial_sog_max < self.initial_sog_min:
            raise ValueError(
                f"initial_sog_max ({self.initial_sog_max}) must be >= initial_sog_min ({self.initial_sog_min})"
            )
        
        # Reward weights (objective function) - can be customized
        self.reward_weights = config.get('reward_weights', {
            'cohesion': 0.1,
            'alignment': 1.0,
            'separation': -0.2,
            'upstream_progress': 10.0,
            'energy_efficiency': 2.0,
            'drafting_benefit': 20.0,
            'boundary_penalty': -10.0,
            'mortality_penalty': -50.0,
            'smoothness_penalty': -0.2,
            'fatigue_penalty': -0.9,
            'stagnation_penalty': -0.9,
            'rheotaxis_alignment': 10.0,
        })
        
        # Training state
        self.best_weights = self.initial_weights
        self.best_reward = -np.inf
        self.episode_history = []  # List of (episode_num, reward)

    def _apply_initial_state_policy(self, sim: Any) -> None:
        """Apply configured initial heading/SOG policy after spatial reset."""
        rng = getattr(sim, "rng", None)
        if rng is None:
            rng = np.random.default_rng()

        heading_mode = self.initial_heading_mode
        if heading_mode in {"environment", "upstream"}:
            pass
        elif heading_mode == "uniform":
            sim.heading = rng.uniform(0.0, 2.0 * np.pi, size=sim.num_agents).astype(np.float32)
        elif heading_mode == "downstream":
            vx = np.asarray(getattr(sim, "x_vel", np.zeros(sim.num_agents)), dtype=float)
            vy = np.asarray(getattr(sim, "y_vel", np.zeros(sim.num_agents)), dtype=float)
            heading = np.arctan2(vy, vx)
            invalid = ~np.isfinite(heading)
            if np.any(invalid):
                heading[invalid] = rng.uniform(0.0, 2.0 * np.pi, size=int(np.count_nonzero(invalid)))
            sim.heading = np.asarray(heading, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported initial_heading_mode: {self.initial_heading_mode}")

        sog_mode = self.initial_sog_mode
        if sog_mode == "environment":
            pass
        elif sog_mode == "uniform":
            sim.sog = rng.uniform(self.initial_sog_min, self.initial_sog_max, size=sim.num_agents).astype(np.float32)
        elif sog_mode == "fixed":
            sim.sog = np.full(sim.num_agents, self.initial_sog_min, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported initial_sog_mode: {self.initial_sog_mode}")

        sim.initial_fish_vel = np.stack(
            (sim.sog * np.cos(sim.heading), sim.sog * np.sin(sim.heading)),
            axis=1,
        ).astype(np.float32)
        
    def run_episode(
        self,
        weights: BehavioralWeights,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        progress_interval: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a single simulation episode with given weights.
        
        Args:
            weights: Behavioral weights to use for this episode
        
        Returns:
            Tuple of (positions, headings, velocities, battery, alive, velocity_field) histories
            Each is (num_timesteps, num_agents, ...) array
        """
        # Create simulation with weights
        sim = self.simulation_factory(weights)
        
        # Reset spatial state to get new random starting positions for this episode
        sim.reset_spatial_state()
        self._apply_initial_state_policy(sim)
        
        # Get number of timesteps from simulation
        num_timesteps = sim.num_timesteps
        num_agents = sim.num_agents
        
        # Allocate arrays for trajectory history
        positions_history = np.zeros((num_timesteps, num_agents, 2), dtype=np.float32)
        headings_history = np.zeros((num_timesteps, num_agents), dtype=np.float32)
        velocities_history = np.zeros((num_timesteps, num_agents, 2), dtype=np.float32)
        battery_history = np.zeros((num_timesteps, num_agents), dtype=np.float32)
        alive_history = np.ones((num_timesteps, num_agents), dtype=bool)
        velocity_field_history = np.zeros((num_timesteps, num_agents, 2), dtype=np.float32)
        
        # Run simulation timesteps (always start from t=0 for each episode)
        progress_interval = max(1, int(progress_interval))
        for t in range(num_timesteps):
            # Run one timestep
            sim.timestep(t, self.dt)
            
            # Collect state
            positions_history[t, :, 0] = sim.X
            positions_history[t, :, 1] = sim.Y
            headings_history[t] = sim.heading
            velocities_history[t, :, 0] = sim.fish_x_vel
            velocities_history[t, :, 1] = sim.fish_y_vel
            battery_history[t] = sim.battery
            alive_history[t] = (sim.dead == 0)
            
            # Collect water velocity field at agent positions (for flow integration)
            velocity_field_history[t, :, 0] = sim.x_vel
            velocity_field_history[t, :, 1] = sim.y_vel

            if progress_callback is not None:
                step_num = t + 1
                if step_num == 1 or step_num == num_timesteps or (step_num % progress_interval == 0):
                    progress_callback(step_num, num_timesteps)
        
        # Clean up simulation
        sim.close()
        
        return positions_history, headings_history, velocities_history, battery_history, alive_history, velocity_field_history
    
    def train(
        self,
        num_episodes: int = 50,
        verbose: bool = True
    ) -> Tuple[BehavioralWeights, list]:
        """
        Train behavioral weights through episodic RL.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Print progress
        
        Returns:
            Tuple of (best_weights, history)
            history is list of (episode, reward) tuples
        """
        if verbose:
            print(f"Starting RL training: {num_episodes} episodes")
            print(f"Exploration noise: {self.exploration_noise}")
            print()
        
        current_weights = self.initial_weights
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Run episode with current weights
            positions, headings, velocities, battery, alive, velocity_field = self.run_episode(current_weights)
            
            # Compute reward using flow vector integration for upstream progress
            reward, components = compute_episode_reward(
                positions, headings, velocities, alive,
                body_length=self.body_length,
                threat_level=current_weights.threat_level,
                battery_history=battery,
                velocity_field_history=velocity_field,
                reward_weights=self.reward_weights  # Pass customizable reward weights
            )
            
            elapsed = time.time() - start_time
            
            # Track best weights
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_weights = current_weights
                improved = True
            else:
                improved = False
            
            # Store history
            self.episode_history.append((episode, float(reward)))
            
            # Progress report
            if verbose:
                status = "✓ NEW BEST" if improved else ""
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"reward={reward:.2f} "
                      f"(best={self.best_reward:.2f}) "
                      f"[{elapsed:.1f}s] {status}")
                if episode % 10 == 0 and episode > 0:
                    print(f"  Components: cohesion={components['cohesion']:.1f}, "
                          f"alignment={components['alignment']:.1f}, "
                          f"upstream={components['upstream_progress']:.1f}")
            
            # Adaptive mutation strategy for next episode
            # Key insight: WORSE solutions should explore MORE aggressively
            # BETTER solutions should refine more carefully
            
            if improved:
                # New best found - small refinement mutation from this new best
                current_weights = current_weights.mutate(mutation_scale=self.exploration_noise * 0.5)
            else:
                # Didn't improve - need to explore more aggressively
                # Compute performance gap to decide mutation strength
                if self.best_reward > -1e6:  # Valid best exists
                    performance_ratio = reward / self.best_reward if self.best_reward > 0 else 0.0
                    
                    # If we're far from best (performance_ratio < 0.5), mutate A LOT
                    # If we're close to best (performance_ratio > 0.8), mutate moderately
                    if performance_ratio < 0.5:
                        # Very poor performance - large exploration from best
                        mutation_scale = self.exploration_noise * 3.0
                    elif performance_ratio < 0.8:
                        # Moderate performance - normal exploration
                        mutation_scale = self.exploration_noise * 1.5
                    else:
                        # Close to best - small refinement
                        mutation_scale = self.exploration_noise
                    
                    # Mutate from best with adaptive scale
                    current_weights = self.best_weights.mutate(mutation_scale=mutation_scale)
                else:
                    # First episode or no valid best yet
                    current_weights = current_weights.mutate(mutation_scale=self.exploration_noise)
        
        if verbose:
            print()
            print(f"Training complete!")
            print(f"Best reward: {self.best_reward:.2f}")
        
        return self.best_weights, self.episode_history
    
    def save_best_weights(self, filepath: Path) -> None:
        """Save best weights to JSON file."""
        self.best_weights.to_json(filepath)
        if hasattr(self, 'episode_history') and self.episode_history:
            # Also save training history
            history_path = filepath.parent / f"{filepath.stem}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.episode_history, f, indent=2)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get summary statistics from training."""
        if not self.episode_history:
            return {}
        
        rewards = [ep['reward'] for ep in self.episode_history]
        
        return {
            'num_episodes': len(self.episode_history),
            'best_reward': self.best_reward,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'final_reward': rewards[-1],
            'improvement': self.best_reward - rewards[0] if len(rewards) > 0 else 0.0
        }
