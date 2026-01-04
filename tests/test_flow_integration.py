"""
Test flow vector integration for upstream progress measurement.

Verifies that flow integration correctly computes upstream distance
for various movement patterns.
"""

import numpy as np
import pytest
from emergent.salmon_abm.rl_training import compute_episode_reward


def test_flow_integration_straight_upstream():
    """
    Test: Fish swimming straight upstream against uniform flow.
    Expected: Upstream progress = distance traveled
    """
    T = 10
    N = 5
    
    # Create histories
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # Uniform downstream flow (flowing in +X direction)
    velocity_field = np.zeros((T, N, 2))
    velocity_field[:, :, 0] = 1.0  # 1 m/s in +X
    velocity_field[:, :, 1] = 0.0
    
    # Fish swim straight upstream (-X direction)
    for t in range(T):
        positions[t, :, 0] = -t * 2.0  # Move 2m/timestep in -X
        positions[t, :, 1] = 0.0
    
    reward, components = compute_episode_reward(
        positions, headings, velocities, alive,
        body_length=0.5,
        battery_history=battery,
        velocity_field_history=velocity_field
    )
    
    upstream_progress = components['upstream_progress'] / 5.0  # Undo 5x weight
    
    # Should be ~2m/timestep upstream progress
    assert upstream_progress > 1.5, f"Expected >1.5 m/timestep, got {upstream_progress}"
    print(f"✓ Straight upstream: {upstream_progress:.2f} m/timestep")


def test_flow_integration_perpendicular():
    """
    Test: Fish swimming perpendicular to flow.
    Expected: Zero upstream progress
    """
    T = 10
    N = 5
    
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # Flow in +X direction
    velocity_field = np.zeros((T, N, 2))
    velocity_field[:, :, 0] = 1.0
    velocity_field[:, :, 1] = 0.0
    
    # Fish move in +Y direction (perpendicular)
    for t in range(T):
        positions[t, :, 0] = 0.0
        positions[t, :, 1] = t * 2.0
    
    reward, components = compute_episode_reward(
        positions, headings, velocities, alive,
        body_length=0.5,
        battery_history=battery,
        velocity_field_history=velocity_field
    )
    
    upstream_progress = components['upstream_progress'] / 5.0
    
    # Should be near zero (perpendicular movement)
    assert abs(upstream_progress) < 0.5, f"Expected ~0, got {upstream_progress}"
    print(f"✓ Perpendicular movement: {upstream_progress:.2f} m/timestep (near zero)")


def test_flow_integration_with_current():
    """
    Test: Fish drifting downstream with current.
    Expected: Negative upstream progress
    """
    T = 10
    N = 5
    
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # Flow in +X direction
    velocity_field = np.zeros((T, N, 2))
    velocity_field[:, :, 0] = 1.0
    velocity_field[:, :, 1] = 0.0
    
    # Fish drift downstream (+X direction)
    for t in range(T):
        positions[t, :, 0] = t * 1.5  # Drift with current
        positions[t, :, 1] = 0.0
    
    reward, components = compute_episode_reward(
        positions, headings, velocities, alive,
        body_length=0.5,
        battery_history=battery,
        velocity_field_history=velocity_field
    )
    
    upstream_progress = components['upstream_progress'] / 5.0
    
    # Should be negative (moving downstream)
    assert upstream_progress < -0.5, f"Expected negative, got {upstream_progress}"
    print(f"✓ Drifting downstream: {upstream_progress:.2f} m/timestep (negative)")


def test_flow_integration_diagonal():
    """
    Test: Fish swimming at 45° angle to flow.
    Expected: Partial upstream progress (cos(45°) ≈ 0.707)
    """
    T = 10
    N = 5
    
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # Flow in +X direction
    velocity_field = np.zeros((T, N, 2))
    velocity_field[:, :, 0] = 1.0
    velocity_field[:, :, 1] = 0.0
    
    # Fish swim at 45° upstream (-X and +Y equally)
    for t in range(T):
        positions[t, :, 0] = -t * 1.0  # Upstream component
        positions[t, :, 1] = t * 1.0   # Lateral component
    
    reward, components = compute_episode_reward(
        positions, headings, velocities, alive,
        body_length=0.5,
        battery_history=battery,
        velocity_field_history=velocity_field
    )
    
    upstream_progress = components['upstream_progress'] / 5.0
    
    # Should be ~0.707 * 1.0 = 0.707 m/timestep (projection)
    assert 0.5 < upstream_progress < 1.2, f"Expected 0.5-1.2, got {upstream_progress}"
    print(f"✓ Diagonal movement (45°): {upstream_progress:.2f} m/timestep (~0.707 expected)")


def test_flow_integration_varying_flow():
    """
    Test: Fish in varying flow velocities.
    Expected: Integration handles changing flow directions
    """
    T = 10
    N = 5
    
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # Varying flow: alternating directions
    velocity_field = np.zeros((T, N, 2))
    for t in range(T):
        if t < 5:
            velocity_field[t, :, 0] = 1.0  # Flow +X
        else:
            velocity_field[t, :, 0] = -1.0  # Flow -X (reversed)
    
    # Fish always move in +X direction
    for t in range(T):
        positions[t, :, 0] = t * 1.0
        positions[t, :, 1] = 0.0
    
    reward, components = compute_episode_reward(
        positions, headings, velocities, alive,
        body_length=0.5,
        battery_history=battery,
        velocity_field_history=velocity_field
    )
    
    upstream_progress = components['upstream_progress'] / 5.0
    
    # First half: moving downstream (negative)
    # Second half: moving upstream (positive)
    # Should roughly cancel out
    assert abs(upstream_progress) < 0.5, f"Expected ~0 (cancellation), got {upstream_progress}"
    print(f"✓ Varying flow: {upstream_progress:.2f} m/timestep (flow reversal handled)")


def test_flow_integration_no_velocity():
    """
    Test: Fish in stagnant water (no flow).
    Expected: Error raised (need velocity field)
    """
    T = 10
    N = 5
    
    positions = np.zeros((T, N, 2))
    headings = np.zeros((T, N))
    velocities = np.zeros((T, N, 2))
    alive = np.ones((T, N), dtype=bool)
    battery = np.ones((T, N))
    
    # No velocity field provided
    with pytest.raises(ValueError, match="velocity_field_history is required"):
        compute_episode_reward(
            positions, headings, velocities, alive,
            body_length=0.5,
            battery_history=battery,
            velocity_field_history=None  # Should fail
        )
    
    print("✓ Error raised when velocity_field_history is None")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Flow Vector Integration for Upstream Progress")
    print("=" * 70)
    print()
    
    test_flow_integration_straight_upstream()
    test_flow_integration_perpendicular()
    test_flow_integration_with_current()
    test_flow_integration_diagonal()
    test_flow_integration_varying_flow()
    test_flow_integration_no_velocity()
    
    print()
    print("=" * 70)
    print("All tests passed! Flow integration working correctly.")
    print("=" * 70)
