import numpy as np
from emergent.fish_passage import merged_kernels
from emergent.salmon_abm import sockeye


def test_drag_and_battery_matches_sockeye():
    rng = np.random.RandomState(2)
    n = 16
    sog = np.abs(rng.randn(n))
    heading = rng.randn(n)
    x_vel = rng.randn(n) * 0.2
    y_vel = rng.randn(n) * 0.2
    mask = rng.rand(n) > 0.1
    density = 1.0
    surface_areas = np.abs(rng.randn(n)) + 0.1
    drag_coeffs = np.abs(rng.randn(n)) + 0.1
    wave_drag = np.abs(rng.randn(n)) + 0.1
    swim_behav = rng.randint(0,5,size=n)
    max_s_U = np.full(n, 1.0)
    max_p_U = np.full(n, 2.0)
    battery = np.clip(np.abs(rng.randn(n)), 0.0, 1.0)
    per_rec = np.abs(rng.randn(n)) * 0.1
    ttf = np.abs(rng.randn(n)) + 0.1
    dt = 0.5
    swim_speeds_buf = np.zeros((n, 4))

    out_new = merged_kernels.drag_and_battery(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, per_rec, ttf, dt, True, swim_speeds_buf)
    out_old = sockeye._drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery.copy(), per_rec, ttf, dt, True)

    for a, b in zip(out_new, out_old):
        if isinstance(a, np.ndarray):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
        else:
            assert a == b
