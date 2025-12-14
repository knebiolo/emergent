import numpy as np
from emergent.fish_passage.simulation.step import step_agents


def test_step_single_agent_basic():
    rng = np.random.RandomState(4)
    n = 8
    sog = np.abs(rng.randn(n))
    heading = rng.randn(n)
    x_vel = rng.randn(n) * 0.05
    y_vel = rng.randn(n) * 0.05
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
    dt = 0.2

    drags, new_batt, ss = step_agents(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, per_rec, ttf, dt)

    assert drags.shape == (n,2)
    assert new_batt.shape == (n,)
    assert ss.shape == (n,)
    assert np.all(new_batt >= 0.0) and np.all(new_batt <= 1.0)
