import numpy as np
from emergent.fish_passage import merged_kernels


def test_integration_step_shapes_and_bounds():
    rng = np.random.RandomState(3)
    n = 10
    sog = np.abs(rng.randn(n))
    heading = rng.randn(n)
    x_vel = rng.randn(n) * 0.1
    y_vel = rng.randn(n) * 0.1
    mask = np.ones(n, dtype=bool)
    density = 1.0
    surface_areas = np.full(n, 1.0)
    drag_coeffs = np.full(n, 0.1)
    wave_drag = np.full(n, 0.1)
    swim_behav = np.zeros(n, dtype=int)
    max_s_U = np.full(n, 1.0)
    max_p_U = np.full(n, 2.0)
    battery = np.ones(n) * 0.5
    per_rec = np.zeros(n)
    ttf = np.full(n, 1e6)
    dt = 0.1

    ss, bl_s, prolonged, sprint, sustained, drags, new_batt = merged_kernels.drag_and_battery(
        sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, per_rec, ttf, dt, True
    )

    assert ss.shape == (n,)
    assert drags.shape == (n,2)
    assert np.all(new_batt >= 0.0) and np.all(new_batt <= 1.0)
