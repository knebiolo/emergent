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
    
    # Drafting (energy-efficient formations)
    drafting_enabled: bool = True  # Enable drafting benefit calculations
    
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
        
        for key, value in data.items():
            if isinstance(value, bool):
                randomized[key] = value
            elif isinstance(value, (int, float)):
                # Apply larger random perturbation for initial exploration
                perturbation = rng.normal(0, abs(value) * scale)
                randomized[key] = max(0.0, value + perturbation)
            else:
                randomized[key] = value
        
        return BehavioralWeights.from_dict(randomized)
    
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
    longitudinal_profile: Optional[Any] = None
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
    T, N = positions_history.shape[0], positions_history.shape[1]
    
    if T == 0 or N == 0:
        return 0.0, {}
    
    # =================================================================
    # 1. Schooling Quality (averaged over time)
    # =================================================================
    cohesion_scores = []
    alignment_scores = []
    separation_penalties = []
    
    for t in range(T):
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
            
            cohesion_scores.append(np.mean(cohesion))
            alignment_scores.append(np.mean(alignment))
            separation_penalties.append(np.mean(separation))
    
    mean_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0
    mean_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
    mean_separation = np.mean(separation_penalties) if separation_penalties else 0.0
    
    # =================================================================
    # 2. Upstream Progress (MUST use longitudinal profile for accurate river distance)
    # =================================================================
    if longitudinal_profile is None:
        raise ValueError("longitudinal_profile is required for computing upstream progress - cannot use Y-displacement fallback")
    
    # Use longitudinal distance along river channel
    # longitudinal_profile is a shapely geometry (LineString) from the shapefile
    from shapely.geometry import Point
    
    initial_pos = np.mean(positions_history[0, alive_history[0]], axis=0)
    final_pos = np.mean(positions_history[-1, alive_history[-1]], axis=0) if np.any(alive_history[-1]) else initial_pos
    
    # Convert to shapely Points and project onto longitudinal line
    initial_point = Point(initial_pos[0], initial_pos[1])
    final_point = Point(final_pos[0], final_pos[1])
    
    # Get distance along the river centerline using shapely's project method
    initial_river_dist = longitudinal_profile.project(initial_point)
    final_river_dist = longitudinal_profile.project(final_point)
    
    total_upstream = final_river_dist - initial_river_dist
    
    mean_upstream_progress = total_upstream / T  # Meters per timestep
    
    # =================================================================
    # 3. Energy Efficiency (distance / speed²)
    # =================================================================
    # Energy ∝ speed², so efficiency = distance traveled / sum(speed²)
    total_distance = 0.0
    total_energy = 0.0
    
    for t in range(1, T):
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
    agents_near_boundary = 0
    if boundary_coords is not None:
        # Count timesteps where agents are near boundary
        for t in range(T):
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
    initial_alive = np.sum(alive_history[0])
    final_alive = np.sum(alive_history[-1])
    dead_count = initial_alive - final_alive
    
    # =================================================================
    # 7. Movement Smoothness (acceleration changes)
    # =================================================================
    # Compute acceleration changes (jerk)
    accel_smoothness_penalty = 0.0
    
    for t in range(2, T):
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
        accel_smoothness_penalty += np.mean(jerk)
    
    accel_smoothness_penalty /= max(1, T - 2)  # Average over timesteps
    
    # =================================================================
    # 8. Fatigue Penalty (CRITICAL for preventing exhaustion)
    # =================================================================
    # Heavily penalize low battery states to incentivize energy management
    fatigue_penalty = 0.0
    if battery_history is not None:
        # Battery is 0.0 (depleted) to 1.0 (full)
        # Penalize time spent below threshold
        low_battery_threshold = 0.3  # Below 30% is critical
        for t in range(T):
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
        
        # Average over timesteps
        fatigue_penalty /= T
    
    # =================================================================
    # Total Reward Calculation
    # =================================================================
    
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
        mean_cohesion * 10.0 +
        (mean_alignment + 1.0) * 5.0 +  # Shift -1:1 → 0:2, scale to 0:10
        mean_separation * 5.0 +
        mean_upstream_progress * 5.0 +  # INCREASED from 0.5 to 5.0 - must make progress!
        energy_efficiency * 2.0 +
        mean_drafting_benefit * 20.0 +
        agents_near_boundary * -5.0 +
        dead_count * -50.0 +
        accel_smoothness_penalty * -0.2 +
        fatigue_penalty * -10.0 +  # CRITICAL: Heavily penalize low battery states
        min_schooling_weight_penalty +  # CRITICAL: Prevent zero schooling weights
        weight_diversity_bonus  # Encourage balanced weight distribution
    )
    
    components = {
        'cohesion': mean_cohesion * 10.0,
        'alignment': (mean_alignment + 1.0) * 5.0,
        'separation': mean_separation * 5.0,
        'upstream_progress': mean_upstream_progress * 5.0,
        'energy_efficiency': energy_efficiency * 2.0,
        'drafting_benefit': mean_drafting_benefit * 20.0,
        'boundary_penalty': agents_near_boundary * -5.0,
        'mortality_penalty': dead_count * -50.0,
        'smoothness_penalty': accel_smoothness_penalty * -0.2,
        'fatigue_penalty': fatigue_penalty * -10.0,
        'min_schooling_penalty': min_schooling_weight_penalty,
        'weight_diversity_bonus': weight_diversity_bonus,
        'total': reward
    }
    
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
        """
        self.simulation_factory = simulation_factory
        self.initial_weights = initial_weights if initial_weights else BehavioralWeights()
        
        # Extract config
        config = config or {}
        self.exploration_noise = config.get('exploration_noise', 0.1)
        self.body_length = config.get('body_length', 0.5)
        self.dt = config.get('dt', 1.0)
        self.num_timesteps = config.get('num_timesteps', 100)
        
        # Training state
        self.best_weights = self.initial_weights
        self.best_reward = -np.inf
        self.episode_history = []  # List of (episode_num, reward)
        
    def run_episode(
        self,
        weights: BehavioralWeights
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a single simulation episode with given weights.
        
        Args:
            weights: Behavioral weights to use for this episode
        
        Returns:
            Tuple of (positions, headings, velocities, battery, alive) histories
            Each is (num_timesteps, num_agents, ...) array
        """
        # Create simulation with weights
        sim = self.simulation_factory(weights)
        
        # Reset spatial state to get new random starting positions for this episode
        sim.reset_spatial_state()
        
        # Get number of timesteps from simulation
        num_timesteps = sim.num_timesteps
        num_agents = sim.num_agents
        
        # Allocate arrays for trajectory history
        positions_history = np.zeros((num_timesteps, num_agents, 2), dtype=np.float32)
        headings_history = np.zeros((num_timesteps, num_agents), dtype=np.float32)
        velocities_history = np.zeros((num_timesteps, num_agents, 2), dtype=np.float32)
        battery_history = np.zeros((num_timesteps, num_agents), dtype=np.float32)
        alive_history = np.ones((num_timesteps, num_agents), dtype=bool)
        
        # Run simulation timesteps (always start from t=0 for each episode)
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
        
        # Clean up simulation
        sim.close()
        
        return positions_history, headings_history, velocities_history, battery_history, alive_history
    
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
            positions, headings, velocities, battery, alive = self.run_episode(current_weights)
            
            # Compute reward
            reward, components = compute_episode_reward(
                positions, headings, velocities, alive,
                body_length=self.body_length,
                threat_level=current_weights.threat_level
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
            
            # Mutate weights for next episode (exploration)
            current_weights = self.best_weights.mutate(mutation_scale=self.exploration_noise)
        
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
