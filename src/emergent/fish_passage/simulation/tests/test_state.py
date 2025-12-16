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
