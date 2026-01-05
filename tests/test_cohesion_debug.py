"""
Test cohesion calculation to diagnose why it might be zero.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from emergent.salmon_abm.rl_training import compute_cohesion_score


def test_perfect_cohesion():
    """Test with agents at ideal spacing."""
    body_length = 0.3  # 300mm fish
    threat_level = 1.0  # High threat
    
    # Ideal distance = 1.0 BL = 0.3m (from formula: 2.0 - threat_level)
    ideal_dist = body_length * (2.0 - threat_level)
    print(f"Ideal distance: {ideal_dist:.3f}m ({ideal_dist/body_length:.1f} BL)")
    
    # Create 5 agents in a line, spaced at ideal distance
    positions = np.array([
        [0.0, 0.0],
        [ideal_dist, 0.0],
        [ideal_dist * 2, 0.0],
        [ideal_dist * 3, 0.0],
        [ideal_dist * 4, 0.0],
    ])
    
    scores = compute_cohesion_score(positions, body_length, threat_level, sensory_range=5.0)
    
    print(f"\nTest 1: Perfect spacing at ideal distance")
    print(f"Positions:\n{positions}")
    print(f"Cohesion scores: {scores}")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Sum: {np.sum(scores):.1f}")
    
    # Expected: High scores for middle agents (have neighbors on both sides)
    # Edge agents might be lower (only one side of neighbors)


def test_tight_school():
    """Test with agents clustered tightly together."""
    body_length = 0.3
    threat_level = 1.0
    
    # Create tight cluster (0.5m radius)
    np.random.seed(42)
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
    radius = 0.5
    positions = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    scores = compute_cohesion_score(positions, body_length, threat_level, sensory_range=5.0)
    
    print(f"\n\nTest 2: Tight circular school (radius={radius}m)")
    print(f"Cohesion scores: {scores[:5]}... (showing first 5)")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Sum: {np.sum(scores):.1f}")


def test_dispersed():
    """Test with agents spread far apart."""
    body_length = 0.3
    threat_level = 1.0
    
    # Create dispersed agents (5m spacing)
    positions = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [0.0, 5.0],
        [5.0, 5.0],
    ])
    
    scores = compute_cohesion_score(positions, body_length, threat_level, sensory_range=5.0)
    
    print(f"\n\nTest 3: Dispersed agents (5m spacing)")
    print(f"Cohesion scores: {scores}")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"Sum: {np.sum(scores):.1f}")
    
    # Expected: Low scores (agents far from ideal distance to centroid)


def test_realistic_school():
    """Test with random positions within 2m radius (realistic school)."""
    body_length = 0.3
    threat_level = 1.0
    
    np.random.seed(123)
    N = 100
    # Random positions within 2m radius
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0, 2.0, N)
    positions = np.column_stack([
        radii * np.cos(angles),
        radii * np.sin(angles)
    ])
    
    scores = compute_cohesion_score(positions, body_length, threat_level, sensory_range=5.0)
    
    print(f"\n\nTest 4: Realistic random school (100 agents in 2m radius)")
    print(f"Score stats: min={np.min(scores):.3f}, mean={np.mean(scores):.3f}, max={np.max(scores):.3f}")
    print(f"Zeros: {np.sum(scores == 0.0)} / {N}")
    print(f"Sum: {np.sum(scores):.1f}")
    
    # Expected over 1000 timesteps: sum ~= 100 agents × 1000 timesteps × 0.3 avg = 30,000
    # Weighted: 30,000 × 0.01 = 300 reward points


def test_distance_response():
    """Test how score varies with distance from ideal."""
    body_length = 0.3
    threat_level = 1.0
    ideal_dist = body_length * (2.0 - threat_level)  # 0.3m
    sigma = 1.0 * body_length  # UPDATED: 0.3m (was 0.15m)
    
    print(f"\n\nTest 5: Score vs. Distance from Ideal")
    print(f"Ideal distance: {ideal_dist:.3f}m, Sigma: {sigma:.3f}m (UPDATED to 1.0 BL)")
    print(f"Distance (BL) | Distance (m) | Score")
    print("-" * 40)
    
    for bl in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        dist = bl * body_length
        # Gaussian formula
        score = np.exp(-0.5 * ((dist - ideal_dist) / sigma)**2)
        print(f"{bl:12.1f} | {dist:12.3f} | {score:.4f}")
    
    print("\nAnalysis with sigma = 1.0 BL:")
    print("- At ideal (1.0 BL = 0.3m): score = 1.000")
    print("- At 2.0 BL (0.6m): score = 0.607 (was 0.135)")
    print("- At 3.0 BL (0.9m): score = 0.135 (was 0.0003)")
    print("- At 5.0 BL (1.5m): score = 0.011 (was ~0)")
    print("Much more forgiving! Fish can deviate from ideal spacing.")



if __name__ == '__main__':
    print("="*60)
    print("COHESION CALCULATION DIAGNOSTIC TESTS")
    print("="*60)
    
    test_perfect_cohesion()
    test_tight_school()
    test_dispersed()
    test_realistic_school()
    test_distance_response()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If cohesion sum is zero in actual training:")
    print("1. Fish might be > 2 BL from local centroid (Gaussian drops to ~0)")
    print("2. No neighbors within 5 BL sensory range (isolated agents)")
    print("3. Check if positions have wrong units or scale")
    print("4. Run viewer with debug prints enabled to see actual values")
