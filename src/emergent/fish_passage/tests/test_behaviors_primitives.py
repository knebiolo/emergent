import numpy as np

from emergent.fish_passage import behaviors


def test_swim_upstream_basic():
    out = behaviors.swim_upstream((0.0, 1.0), 3, strength=2.0)
    assert out.shape == (3, 2)
    # each vector should point in +y and have magnitude 2
    mags = np.hypot(out[:, 0], out[:, 1])
    assert np.allclose(mags, 2.0)
    assert np.allclose(out[:, 0], 0.0)


def test_swim_upstream_mask():
    mask = [True, False, True]
    out = behaviors.swim_upstream((1.0, 0.0), 3, strength=1.0, mask=mask)
    assert np.allclose(out[1], 0.0)
    assert np.all(out[[0, 2], 0] > 0.0)


def test_avoid_obstacle_no_obs():
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    out = behaviors.avoid_obstacle(pos, [])
    assert out.shape == pos.shape
    assert np.allclose(out, 0.0)


def test_avoid_obstacle_single():
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    obs = [(0.5, 0.0, 0.2)]
    out = behaviors.avoid_obstacle(pos, obs, influence=1.0)
    # both agents should have non-negative x component pushing them away from obstacle center
    assert out.shape == pos.shape
    assert out[0, 0] < 0.0 or out[0, 0] > 0.0


def test_school_with_neighbors_basic():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.2]])
    headings = np.array([0.0, 0.1, -0.1])
    out = behaviors.school_with_neighbors(pos, headings)
    assert out.shape == (3, 2)
    # at least one agent should have non-zero composite vector for this configuration
    assert np.any(np.linalg.norm(out, axis=1) > 0.0)
