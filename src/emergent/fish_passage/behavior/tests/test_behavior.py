import numpy as np

from emergent.fish_passage.behavior.primitives import schooling_vector, collision_avoidance_vector
from emergent.fish_passage.behavior.manager import BehaviorManager


def test_schooling_vector_simple():
    positions = np.array([[0.0, 0.0], [2.0, 0.0]])
    vecs = schooling_vector(positions, radii=0.0, ideal_dist=1.0)
    # symmetry: agents should have opposite vectors
    assert np.allclose(vecs[0], -vecs[1])


def test_collision_avoidance_vector_simple():
    positions = np.array([[0.0, 0.0], [0.4, 0.0]])
    vecs = collision_avoidance_vector(positions, min_sep=1.0)
    # agent 0 should be repelled from agent 1 (positive x), agent1 repelled negative x
    assert vecs[0][0] < 0 or vecs[1][0] > 0


def test_behavior_manager_step():
    positions = np.array([[0.0, 0.0], [0.8, 0.0], [2.0, 0.0]])
    headings = np.zeros((3,))
    bm = BehaviorManager({'schooling': 1.0, 'collision': 1.0})
    out = bm.step(positions, headings, dt=1.0)
    assert 'desired_heading_vec' in out and 'desired_speed' in out
    assert out['desired_heading_vec'].shape == positions.shape
    assert out['desired_speed'].shape[0] == positions.shape[0]
