import numpy as np

def test_swim_core_numpy_basic():
    from emergent.fish_passage.movement import swim_core
    # deterministic inputs
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    headings = np.array([0.0, np.pi/2])
    speeds = np.array([1.0, 2.0])
    new_pos, new_spd = swim_core(positions, headings, speeds, env_forces=None, dt=1.0)
    assert new_spd.shape == speeds.shape
    # expected displacement: first moves +1 on x, second moves +2 on y
    assert np.allclose(new_pos[0], [1.0, 0.0])
    assert np.allclose(new_pos[1], [1.0, 3.0])
