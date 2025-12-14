import numpy as np
from emergent.fish_passage import swim_speeds
from emergent.salmon_abm import sockeye


def test_swim_speeds_matches_sockeye():
    rng = np.random.RandomState(2)
    n = 50
    x_vel = rng.randn(n) * 0.1
    y_vel = rng.randn(n) * 0.1
    sog = np.abs(rng.randn(n))
    heading = rng.randn(n)

    out_new = swim_speeds.swim_speeds(x_vel, y_vel, sog, heading)
    out_old = sockeye._swim_speeds_numba(x_vel, y_vel, sog, heading)

    assert out_new.shape == out_old.shape
    np.testing.assert_allclose(out_new, out_old, rtol=1e-7, atol=1e-12)
