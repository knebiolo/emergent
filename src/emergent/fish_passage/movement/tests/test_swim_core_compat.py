import numpy as np

def test_swim_core_compat_returns_shapes():
    from emergent.fish_passage.movement.swim_core_compat import _swim_core_numba
    fv0x = np.array([1.0, 0.0])
    fv0y = np.array([0.0, 1.0])
    accx = np.array([0.0, 0.0])
    accy = np.array([0.0, 0.0])
    pidx = np.array([0, 1])
    pidy = np.array([0, 1])
    tired_mask = np.zeros(2, dtype=bool)
    dead_mask = np.zeros(2, dtype=bool)
    mask = np.ones(2, dtype=bool)
    pos, speeds = _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, 1.0)
    assert pos.shape == (2, 2)
    assert speeds.shape == (2,)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(speeds))
