import numpy as np
from emergent.fish_passage.simulation.state import SimulationState
from emergent.fish_passage.simulation import kernels


def test_compute_step_basic():
    s = SimulationState.allocate(4)
    # set simple headings and sog
    s.heading[:] = 0.0
    s.sog[:] = np.array([0.5, 0.6, 0.7, 0.2])

    env = {
        'wx': np.zeros(4),
        'wy': np.zeros(4),
        'density': 1.0,
    }

    out = kernels.compute_step(s, env, dt=1.0)
    assert 'drags' in out and out['drags'].shape == (4, 2)
    assert 'swim_speeds' in out and out['swim_speeds'].shape == (4,)
    assert 'battery' in out and out['battery'].shape == (4,)

    # battery should remain within [0,1]
    assert np.all(out['battery'] >= 0.0)
    assert np.all(out['battery'] <= 1.0)
