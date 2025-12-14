import numpy as np
from emergent.fish_passage import drags
from emergent.salmon_abm import sockeye


def test_compute_drags_matches_sockeye():
    rng = np.random.RandomState(0)
    n = 10
    fx = rng.randn(n)
    fy = rng.randn(n)
    wx = rng.randn(n) * 0.1
    wy = rng.randn(n) * 0.1
    mask = rng.rand(n) > 0.2
    density = 1.0
    surface_areas = np.abs(rng.randn(n)) + 0.1
    drag_coeffs = np.abs(rng.randn(n)) + 0.1
    wave_drag = np.abs(rng.randn(n)) + 0.1
    swim_behav = rng.randint(0, 5, size=n)

    out_new = drags.compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
    out_old = sockeye.compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)

    assert out_new.shape == out_old.shape
    # numeric closeness within reasonable tolerance
    np.testing.assert_allclose(out_new, out_old, rtol=1e-6, atol=1e-8)
