"""
Unit tests for RL training infrastructure.

Tests BehavioralWeights dataclass, metrics, reward function, and RLTrainer.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from emergent.salmon_abm.rl_training import BehavioralWeights


class TestBehavioralWeights:
    """Test BehavioralWeights dataclass."""
    
    def test_default_initialization(self):
        """Test creation with default values."""
        weights = BehavioralWeights()
        
        assert weights.cohesion_weight == 1000.0
        assert weights.alignment_weight == 25000.0
        assert weights.collision_weight == 2000.0
        assert weights.sensory_range == 2.0  # Biological constant
        assert weights.threat_level == 0.3
    
    def test_custom_initialization(self):
        """Test creation with custom values."""
        weights = BehavioralWeights(
            cohesion_weight=5000.0,
            alignment_weight=30000.0,
            threat_level=0.5
        )
        
        assert weights.cohesion_weight == 5000.0
        assert weights.alignment_weight == 30000.0
        assert weights.threat_level == 0.5
    
    def test_to_dict(self):
        """Test dictionary serialization."""
        weights = BehavioralWeights(cohesion_weight=1500.0)
        data = weights.to_dict()
        
        assert isinstance(data, dict)
        assert data['cohesion_weight'] == 1500.0
        assert 'alignment_weight' in data
        assert 'sensory_range' in data
    
    def test_from_dict(self):
        """Test dictionary deserialization."""
        data = {
            'cohesion_weight': 2000.0,
            'alignment_weight': 28000.0,
            'separation_weight': 6000.0,
            'separation_radius': 1.2,
            'rheotaxis_weight': 30000.0,
            'border_cue_weight': 250000.0,
            'border_threshold_multiplier': 2.5,
            'border_max_force': 12.0,
            'collision_weight': 2500.0,
            'collision_radius': 0.6,
            'low_speed_weight': 1800.0,
            'wave_drag_weight': 100.0,
            'refugia_weight': 60000.0,
            'shallow_weight': 600000.0,
            'avoid_weight': 30000.0,
            'sensory_range': 2.5,
            'threat_level': 0.4,
            'cohesion_radius_relaxed': 3.2,
            'cohesion_radius_threatened': 1.6,
        }
        
        weights = BehavioralWeights.from_dict(data)
        
        assert weights.cohesion_weight == 2000.0
        assert weights.alignment_weight == 28000.0
        assert weights.threat_level == 0.4
    
    def test_json_serialization(self, tmp_path):
        """Test JSON save/load round-trip."""
        weights = BehavioralWeights(
            cohesion_weight=1200.0,
            alignment_weight=26000.0,
            threat_level=0.35
        )
        
        json_path = tmp_path / "test_weights.json"
        weights.to_json(json_path)
        
        # Verify file exists and is valid JSON
        assert json_path.exists()
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data['cohesion_weight'] == 1200.0
        
        # Load and verify
        loaded = BehavioralWeights.from_json(json_path)
        assert loaded.cohesion_weight == 1200.0
        assert loaded.alignment_weight == 26000.0
        assert loaded.threat_level == 0.35
    
    def test_json_file_not_found(self, tmp_path):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Behavioral weights file not found"):
            BehavioralWeights.from_json(tmp_path / "nonexistent.json")
    
    def test_to_test_weights_dict(self):
        """Test conversion to simulation.test_weights format."""
        weights = BehavioralWeights(
            rheotaxis_weight=25000.0,
            alignment_weight=20500.0,
            cohesion_weight=11000.0,
            collision_weight=1000.0,
        )
        
        test_weights = weights.to_test_weights_dict()
        
        assert test_weights['rheotaxis'] == 25000.0
        assert test_weights['alignment'] == 20500.0
        assert test_weights['cohesion'] == 11000.0
        assert test_weights['collision'] == 1000.0
        assert 'border' in test_weights
        assert 'shallow' in test_weights
    
    def test_validate_success(self):
        """Test validation passes for valid weights."""
        weights = BehavioralWeights()
        weights.validate()  # Should not raise
    
    def test_validate_negative_weight(self):
        """Test validation fails for negative weights."""
        weights = BehavioralWeights(cohesion_weight=-100.0)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            weights.validate()
    
    def test_validate_sensory_range(self):
        """Test validation of sensory range."""
        # Too small
        weights = BehavioralWeights(sensory_range=0.1)
        with pytest.raises(ValueError, match="sensory_range should be 0.5-5.0"):
            weights.validate()
        
        # Too large
        weights = BehavioralWeights(sensory_range=10.0)
        with pytest.raises(ValueError, match="sensory_range should be 0.5-5.0"):
            weights.validate()
    
    def test_validate_threat_level(self):
        """Test validation of threat level."""
        # Too low
        weights = BehavioralWeights(threat_level=-0.1)
        with pytest.raises(ValueError, match="threat_level must be 0.0-1.0"):
            weights.validate()
        
        # Too high
        weights = BehavioralWeights(threat_level=1.5)
        with pytest.raises(ValueError, match="threat_level must be 0.0-1.0"):
            weights.validate()
    
    def test_validate_cohesion_radii(self):
        """Test validation of cohesion radius relationship."""
        weights = BehavioralWeights(
            cohesion_radius_relaxed=1.0,
            cohesion_radius_threatened=2.0  # Invalid: threatened > relaxed
        )
        
        with pytest.raises(ValueError, match="cohesion_radius_relaxed.*must be >="):
            weights.validate()
    
    def test_mutate(self):
        """Test weight mutation for RL exploration."""
        rng = np.random.default_rng(42)  # Deterministic
        weights = BehavioralWeights(cohesion_weight=1000.0)
        
        mutated = weights.mutate(mutation_scale=0.1, rng=rng)
        
        # Original unchanged
        assert weights.cohesion_weight == 1000.0
        
        # Mutated is different (with high probability)
        assert mutated.cohesion_weight != 1000.0
        
        # Mutated is non-negative
        assert mutated.cohesion_weight >= 0.0
        
        # Mutated is within reasonable range (10% scale)
        assert 500.0 < mutated.cohesion_weight < 1500.0
    
    def test_mutate_preserves_non_negativity(self):
        """Test mutation clips to non-negative values."""
        rng = np.random.default_rng(42)
        
        # Run multiple mutations to test clipping
        weights = BehavioralWeights(cohesion_weight=100.0)
        
        for _ in range(10):
            mutated = weights.mutate(mutation_scale=0.5, rng=rng)
            data = mutated.to_dict()
            
            # All weights should be non-negative
            for key, value in data.items():
                assert value >= 0.0, f"{key} became negative: {value}"
    
    def test_mutate_deterministic_with_seed(self):
        """Test mutation is deterministic with same seed."""
        weights = BehavioralWeights()
        
        rng1 = np.random.default_rng(123)
        mutated1 = weights.mutate(rng=rng1)
        
        rng2 = np.random.default_rng(123)
        mutated2 = weights.mutate(rng=rng2)
        
        # Same seed -> same mutation
        assert mutated1.cohesion_weight == mutated2.cohesion_weight
        assert mutated1.alignment_weight == mutated2.alignment_weight
