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
from typing import Dict, Optional
from pathlib import Path


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
