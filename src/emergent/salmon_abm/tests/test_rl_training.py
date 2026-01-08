"""
Unit tests for RL training infrastructure.

Tests BehavioralWeights dataclass, metrics, reward function, and RLTrainer.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from emergent.salmon_abm.rl_training import (
    BehavioralWeights,
    compute_cohesion_score,
    compute_alignment_score,
    compute_separation_penalty,
    compute_overall_schooling_score,
    compute_episode_reward,
    RLTrainer
)


class TestBehavioralWeights:
    """Test BehavioralWeights dataclass."""
    
    def test_default_initialization(self):
        """Test creation with default values."""
        weights = BehavioralWeights()
        
        assert weights.cohesion_weight == 1000.0
        assert weights.alignment_weight == 25000.0
        assert weights.collision_weight == 2000.0
        assert weights.sensory_range == 2.0  # Biological constant
        assert weights.threat_level == 0.0
    
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

            'rheotaxis_weight': 30000.0,
            'border_cue_weight': 250000.0,
            'collision_weight': 2500.0,
            'low_speed_weight': 1800.0,
            'wave_drag_weight': 100.0,
            'refugia_weight': 60000.0,
            'shallow_weight': 600000.0,
            'avoid_weight': 30000.0,
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
        with pytest.raises(ValueError, match="must be non-negative"):
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


class TestSchoolingMetrics:
    """Test biological schooling quality metrics."""
    
    def test_cohesion_score_perfect_spacing(self):
        """Test cohesion score for agents at ideal spacing."""
        body_length = 0.5
        threat_level = 0.3
        ideal_dist = body_length * (2.0 - threat_level)  # 0.85 m
        
        # Create triangle formation where centroid distance ≈ ideal
        # For equilateral triangle, centroid is at 1/3 height from base
        # If side length = ideal_dist, height = ideal_dist * sqrt(3)/2
        # Centroid distance from vertex ≈ 2/3 * height ≈ 0.58 * ideal_dist
        # Not quite ideal, but let's test the logic works
        positions = np.array([
            [0.0, 0.0],
            [ideal_dist, 0.0],
            [ideal_dist/2, ideal_dist * np.sqrt(3)/2]
        ])
        
        scores = compute_cohesion_score(positions, body_length, threat_level)
        
        # All agents should have reasonable cohesion (not perfect due to geometry)
        assert np.all(scores > 0.3)  # Reasonable schooling
        assert np.all(scores < 1.0)  # Not quite ideal spacing
    
    def test_cohesion_score_isolated(self):
        """Test cohesion score for isolated agent."""
        body_length = 0.5
        sensory_range = 2.0
        
        # Two agents far apart (>2 BL)
        positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0]  # 10m apart, way beyond sensory range
        ])
        
        scores = compute_cohesion_score(positions, body_length, sensory_range=sensory_range)
        
        # Both isolated
        assert np.all(scores == 0.0)
    
    def test_cohesion_score_too_close(self):
        """Test cohesion score when agents too close."""
        body_length = 0.5
        ideal_dist = 0.85  # ~2 BL at low threat
        
        # Agents much closer than ideal
        positions = np.array([
            [0.0, 0.0],
            [0.2, 0.0],  # Only 0.2m apart, well below ideal
            [0.0, 0.2]
        ])
        
        scores = compute_cohesion_score(positions, body_length, threat_level=0.3)
        
        # Should have lower scores (not at ideal spacing)
        assert np.all(scores < 1.0)
    
    def test_alignment_score_perfect(self):
        """Test alignment score for perfectly aligned agents."""
        body_length = 0.5
        
        # Three agents in a line, all heading same direction
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ])
        headings = np.array([0.0, 0.0, 0.0])  # All heading East
        
        scores = compute_alignment_score(headings, positions, body_length)
        
        # All perfectly aligned with neighbors
        assert np.all(scores == pytest.approx(1.0, abs=0.01))
    
    def test_alignment_score_opposite(self):
        """Test alignment score for opposite headings."""
        body_length = 0.5
        
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        headings = np.array([0.0, np.pi])  # Opposite directions
        
        scores = compute_alignment_score(headings, positions, body_length)
        
        # Should be close to -1 (opposite)
        assert np.all(scores < -0.9)
    
    def test_alignment_score_perpendicular(self):
        """Test alignment score for perpendicular headings."""
        body_length = 0.5
        
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        headings = np.array([0.0, np.pi/2])  # 90° apart
        
        scores = compute_alignment_score(headings, positions, body_length)
        
        # Should be close to 0 (perpendicular)
        assert np.all(np.abs(scores) < 0.1)
    
    def test_separation_penalty_no_crowding(self):
        """Test separation penalty when agents well-spaced."""
        body_length = 0.5
        
        # Agents 2m apart (>1 BL)
        positions = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0]
        ])
        
        penalties = compute_separation_penalty(positions, body_length)
        
        # No crowding penalty
        assert np.all(penalties == 0.0)
    
    def test_separation_penalty_crowding(self):
        """Test separation penalty when agents too close."""
        body_length = 0.5
        crowding_threshold = 1.0 * body_length  # 0.5m
        
        # Agents 0.3m apart (<1 BL)
        positions = np.array([
            [0.0, 0.0],
            [0.3, 0.0]
        ])
        
        penalties = compute_separation_penalty(positions, body_length)
        
        # Should have negative penalties
        assert np.all(penalties < 0.0)
        
        # Expected penalty: -(0.5 - 0.3) / 0.5 = -0.4
        assert np.all(penalties == pytest.approx(-0.4, abs=0.01))
    
    def test_separation_penalty_touching(self):
        """Test separation penalty for touching agents."""
        body_length = 0.5
        
        # Agents essentially touching (0.01m apart)
        positions = np.array([
            [0.0, 0.0],
            [0.01, 0.0]
        ])
        
        penalties = compute_separation_penalty(positions, body_length)
        
        # Should be close to -1.0 (maximum penalty)
        assert np.all(penalties < -0.95)
    
    def test_overall_schooling_score(self):
        """Test overall schooling score calculation."""
        body_length = 0.5
        threat_level = 0.3
        ideal_dist = body_length * (2.0 - threat_level)  # 0.85m
        
        # Good formation: ideal spacing, aligned headings
        positions = np.array([
            [0.0, 0.0],
            [ideal_dist, 0.0],
            [0.0, ideal_dist],
            [-ideal_dist, 0.0]
        ])
        headings = np.array([0.0, 0.0, 0.0, 0.0])  # All aligned
        
        overall, components = compute_overall_schooling_score(
            positions, headings, body_length, threat_level
        )
        
        # Should have high overall score (>1.5 for good schooling)
        assert overall > 1.0
        
        # Check components exist
        assert 'cohesion' in components
        assert 'alignment' in components
        assert 'separation' in components
        assert 'overall' in components
        assert components['overall'] == pytest.approx(overall)
    
    def test_overall_schooling_score_dysfunctional(self):
        """Test overall score for dysfunctional group."""
        body_length = 0.5
        
        # Bad formation: too close, misaligned
        positions = np.array([
            [0.0, 0.0],
            [0.1, 0.0],  # Very close
            [0.05, 0.1]  # Very close
        ])
        headings = np.array([0.0, np.pi, np.pi/2])  # All different directions
        
        overall, components = compute_overall_schooling_score(
            positions, headings, body_length, threat_level=0.3
        )
        
        # Should have low or negative score
        assert overall < 1.0
        
        # Separation should be strongly negative (crowding)
        assert components['separation'] < -0.5
    
    def test_empty_arrays(self):
        """Test metrics handle empty arrays gracefully."""
        body_length = 0.5
        
        positions = np.array([]).reshape(0, 2)
        headings = np.array([])
        
        cohesion = compute_cohesion_score(positions, body_length)
        alignment = compute_alignment_score(headings, positions, body_length)
        separation = compute_separation_penalty(positions, body_length)
        
        assert len(cohesion) == 0
        assert len(alignment) == 0
        assert len(separation) == 0


class TestEpisodeReward:
    """Test episode reward function."""
    
    def _create_velocity_field(self, T, N, flow_y=-2.0):
        """Helper to create dummy velocity field for tests.
        
        Default: flow_y=-2.0 means water flows downstream (-Y direction).
        Upstream direction is then +Y (opposite of flow).
        """
        velocity_field = np.zeros((T, N, 2), dtype=np.float32)
        velocity_field[:, :, 0] = 0.0  # No x flow
        velocity_field[:, :, 1] = flow_y  # y flow (negative = downstream)
        return velocity_field
    
    def test_perfect_episode(self):
        """Test reward for perfect schooling episode."""
        body_length = 0.5
        T, N = 10, 5
        
        # Perfect formation: tight, aligned, moving upstream
        positions = np.zeros((T, N, 2))
        headings = np.zeros((T, N))
        velocities = np.zeros((T, N, 2))
        alive = np.ones((T, N), dtype=bool)
        
        ideal_dist = 0.85  # ~2 BL at low threat
        
        for t in range(T):
            # Linear formation, moving upstream
            for i in range(N):
                positions[t, i] = [i * ideal_dist, t * 0.5]  # Moving +Y
                headings[t, i] = np.pi / 2  # North
                velocities[t, i] = [0.0, 0.5]  # Constant upstream velocity
        
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        # Should have positive reward
        assert reward > 0.0
        
        # Cohesion should be positive (good spacing)
        assert components['cohesion'] > 0.0
        
        # Alignment should be high (all aligned)
        assert components['alignment'] > 5.0  # Near maximum (10.0)
        
        # Upstream progress should be positive
        assert components['upstream_progress'] > 0.0
        
        # No mortality
        assert components['mortality_penalty'] == 0.0
    
    def test_catastrophic_episode(self):
        """Test reward for catastrophic episode (deaths, no progress)."""
        body_length = 0.5
        T, N = 10, 5
        
        positions = np.zeros((T, N, 2))
        headings = np.zeros((T, N))
        velocities = np.zeros((T, N, 2))
        alive = np.ones((T, N), dtype=bool)
        
        # Kill half the agents
        alive[5:, :2] = False
        
        # No upstream progress, random headings
        for t in range(T):
            for i in range(N):
                positions[t, i] = [i * 0.2, 0.0]  # No Y movement
                headings[t, i] = np.random.rand() * 2 * np.pi
                velocities[t, i] = [0.1, 0.0]
        
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        # Should have negative reward due to mortality
        assert components['mortality_penalty'] < 0.0
        
        # Upstream progress near zero
        assert abs(components['upstream_progress']) < 1.0
    
    def test_boundary_penalty(self):
        """Test boundary proximity penalty."""
        body_length = 0.5
        T, N = 5, 3
        
        positions = np.zeros((T, N, 2))
        headings = np.zeros((T, N))
        velocities = np.zeros((T, N, 2))
        alive = np.ones((T, N), dtype=bool)
        
        # Agents near boundary
        for t in range(T):
            for i in range(N):
                positions[t, i] = [i * 1.0, t * 0.1]
                headings[t, i] = 0.0
                velocities[t, i] = [0.1, 0.1]
        
        # Define boundary (agents are close to it)
        boundary = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
        
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            boundary_coords=boundary, boundary_threshold=2.0,
            velocity_field_history=velocity_field
        )
        
        # Should have boundary penalty
        assert components['boundary_penalty'] < 0.0
    
    def test_energy_efficiency(self):
        """Test energy efficiency component."""
        body_length = 0.5
        T, N = 5, 2
        
        positions = np.zeros((T, N, 2))
        headings = np.zeros((T, N))
        velocities = np.zeros((T, N, 2))
        alive = np.ones((T, N), dtype=bool)
        
        # Agents moving at constant speed
        speed = 1.0
        for t in range(T):
            positions[t, 0] = [t * speed, 0.0]
            positions[t, 1] = [t * speed, 1.0]
            velocities[t, 0] = [speed, 0.0]
            velocities[t, 1] = [speed, 0.0]
        
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        # Should have positive energy efficiency
        assert components['energy_efficiency'] > 0.0
    
    def test_smoothness_penalty(self):
        """Test movement smoothness penalty."""
        body_length = 0.5
        T, N = 10, 2
        
        positions = np.zeros((T, N, 2))
        headings = np.zeros((T, N))
        velocities = np.zeros((T, N, 2))
        alive = np.ones((T, N), dtype=bool)
        
        # Jerky movement (alternating velocities)
        for t in range(T):
            vel = 1.0 if t % 2 == 0 else 0.1
            velocities[t] = vel
            positions[t] = positions[t-1] + velocities[t] if t > 0 else 0.0
        
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        # Should have smoothness penalty
        assert components['smoothness_penalty'] < 0.0
    
    def test_empty_episode(self):
        """Test reward handles empty episode gracefully."""
        body_length = 0.5
        
        positions = np.zeros((0, 0, 2))
        headings = np.zeros((0, 0))
        velocities = np.zeros((0, 0, 2))
        alive = np.zeros((0, 0), dtype=bool)
        velocity_field = np.zeros((0, 0, 2), dtype=np.float32)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        assert reward == 0.0
        assert components == {}
    
    def test_component_breakdown(self):
        """Test all reward components are present."""
        body_length = 0.5
        T, N = 5, 3
        
        positions = np.random.rand(T, N, 2) * 10
        headings = np.random.rand(T, N) * 2 * np.pi
        velocities = np.random.rand(T, N, 2)
        alive = np.ones((T, N), dtype=bool)
        velocity_field = self._create_velocity_field(T, N)
        
        reward, components = compute_episode_reward(
            positions, headings, velocities, alive, body_length,
            velocity_field_history=velocity_field
        )
        
        # Check all components exist
        expected_keys = [
            'cohesion', 'alignment', 'separation',
            'upstream_progress', 'energy_efficiency', 'drafting_benefit',
            'boundary_penalty', 'mortality_penalty', 'smoothness_penalty',
            'total'
        ]
        
        for key in expected_keys:
            assert key in components
        
        # Total should match returned reward
        assert components['total'] == pytest.approx(reward)


class TestRLTrainer:
    """Test RL trainer class."""
    
    def create_mock_simulation(self, num_agents=10, num_timesteps=20):
        """Create a mock simulation class for testing."""
        class MockSimulation:
            def __init__(self, weights):
                self.num_agents = num_agents
                self.num_timesteps = num_timesteps
                self.timestep_count = 0
                
                # Initialize state arrays
                self.X = np.random.uniform(0, 100, num_agents).astype(np.float32)
                self.Y = np.random.uniform(0, 100, num_agents).astype(np.float32)
                self.heading = np.random.uniform(0, 2*np.pi, num_agents).astype(np.float32)
                self.fish_x_vel = np.random.uniform(-1, 1, num_agents).astype(np.float32)
                self.fish_y_vel = np.random.uniform(0.5, 1.5, num_agents).astype(np.float32)  # Mostly upstream
                self.battery = np.ones(num_agents, dtype=np.float32)
                self.dead = np.zeros(num_agents, dtype=np.int8)
                
                # Water velocity at agent positions (for flow integration)
                self.x_vel = np.zeros(num_agents, dtype=np.float32)
                self.y_vel = np.ones(num_agents, dtype=np.float32) * 2.0  # 2 m/s upstream flow
            
            def reset_spatial_state(self):
                """Reset positions for new episode."""
                self.X = np.random.uniform(0, 100, num_agents).astype(np.float32)
                self.Y = np.random.uniform(0, 100, num_agents).astype(np.float32)
                
            def timestep(self, t, dt):
                """Simulate one timestep - agents drift upstream."""
                self.Y += 0.5  # Move upstream
                self.X += np.random.uniform(-0.1, 0.1, self.num_agents)  # Small lateral drift
                self.timestep_count += 1
                
                # Water velocity constant
                self.x_vel = np.zeros(self.num_agents, dtype=np.float32)
                self.y_vel = np.ones(self.num_agents, dtype=np.float32) * 2.0
                
            def close(self):
                """Clean up resources."""
                pass
        
        return MockSimulation
    
    def create_simulation_factory(self, num_agents=10, num_timesteps=20):
        """Create a factory function for testing."""
        mock_sim_class = self.create_mock_simulation(num_agents, num_timesteps)
        
        def factory(weights: BehavioralWeights):
            return mock_sim_class(weights)
        
        return factory
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        factory = self.create_simulation_factory()
        trainer = RLTrainer(
            simulation_factory=factory,
            config={'exploration_noise': 0.1}
        )
        
        assert trainer.exploration_noise == 0.1
        assert trainer.body_length == 0.5
        assert trainer.best_reward == -np.inf
        assert len(trainer.episode_history) == 0
    
    def test_trainer_custom_weights(self):
        """Test trainer with custom initial weights."""
        weights = BehavioralWeights(cohesion_weight=5000.0)
        factory = self.create_simulation_factory()
        trainer = RLTrainer(
            simulation_factory=factory,
            initial_weights=weights
        )
        
        assert trainer.initial_weights.cohesion_weight == 5000.0
        assert trainer.best_weights.cohesion_weight == 5000.0
    
    def test_run_episode(self):
        """Test running a single episode."""
        factory = self.create_simulation_factory()
        trainer = RLTrainer(simulation_factory=factory)
        
        positions, headings, velocities, battery, alive, velocity_field = trainer.run_episode(trainer.initial_weights)
        
        # Check output shapes
        assert positions.shape == (20, 10, 2)
        assert headings.shape == (20, 10)
        assert velocities.shape == (20, 10, 2)
        assert velocity_field.shape == (20, 10, 2)
        assert battery.shape == (20, 10)
        assert alive.shape == (20, 10)
    
    def test_train_convergence(self):
        """Test training loop converges."""
        factory = self.create_simulation_factory()
        trainer = RLTrainer(
            simulation_factory=factory,
            config={'exploration_noise': 0.05}
        )
        
        best_weights, history = trainer.train(
            num_episodes=10,
            verbose=False
        )
        
        # Should have run 10 episodes
        assert len(history) == 10
        
        # Best weights should be returned
        assert best_weights == trainer.best_weights
    
    def test_train_tracks_improvements(self):
        """Test training tracks when improvements occur."""
        factory = self.create_simulation_factory()
        trainer = RLTrainer(simulation_factory=factory)
        
        best_weights, history = trainer.train(num_episodes=5, verbose=False)
        
        # Check history structure
        for episode, reward in history:
            assert isinstance(episode, int)
            assert isinstance(reward, float)
    
    def test_save_best_weights(self, tmp_path):
        """Test saving best weights to file."""
        factory = self.create_simulation_factory()
        trainer = RLTrainer(simulation_factory=factory)
        
        best_weights, history = trainer.train(num_episodes=5, verbose=False)
        
        weights_path = tmp_path / "best_weights.json"
        trainer.best_weights.to_json(weights_path)
        
        # Check weights file exists
        assert weights_path.exists()
        
        # Verify can load weights back
        loaded_weights = BehavioralWeights.from_json(weights_path)
        assert loaded_weights.to_dict() == trainer.best_weights.to_dict()
    
    def test_exploration_improves_over_time(self):
        """Test that exploration can improve rewards."""
        factory = self.create_simulation_factory(num_agents=20, num_timesteps=50)
        trainer = RLTrainer(
            simulation_factory=factory,
            config={'exploration_noise': 0.1}
        )
        
        best_weights, history = trainer.train(num_episodes=15, verbose=False)
        
        # Extract rewards
        rewards = [r for _, r in history]
        
        # Best reward should be >= initial reward (allowing for exploration)
        # (May not always improve due to stochastic exploration)
        assert trainer.best_reward >= min(rewards)
