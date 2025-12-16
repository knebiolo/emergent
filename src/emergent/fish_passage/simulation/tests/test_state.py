import numpy as np
from emergent.fish_passage.simulation.state import SimulationState


def test_allocate_shapes_and_defaults():
    s = SimulationState.allocate(5, init_X=1.0, init_Y=2.0)
    assert s.n_agents == 5
    assert s.X.shape == (5,)
    assert s.Y.shape == (5,)
    assert np.allclose(s.X, 1.0)
    assert np.allclose(s.Y, 2.0)
    assert np.allclose(s.prev_X, s.X)
    assert np.all(s.battery == 1.0)
    assert np.all(s.dead == 0)


def test_expanded_fields_and_helpers():
    s = SimulationState.allocate(3)
    assert s.length.shape == (3,)
    assert s.weight.shape == (3,)
    assert s.body_depth.shape == (3,)
    assert s.surface_area.shape == (3,)

    # test update_prev_positions
    s.X[0] = 10.0
    s.Y[0] = -5.0
    s.update_prev_positions()
    assert s.prev_X[0] == 10.0
    assert s.prev_Y[0] == -5.0

    # randomize lengths reproducibly
    s.set_random_lengths(mean=0.3, std=0.01, seed=42)
    m = float(np.mean(s.length))
    assert 0.28 < m < 0.32

    summ = s.summary()
    assert summ['n_agents'] == 3
    assert 'mean_length' in summ
