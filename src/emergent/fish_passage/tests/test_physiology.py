import numpy as np
from emergent.fish_passage.physiology import compute_drags, assess_fatigue_core


def test_compute_drags_zero_mask():
    N = 3
    fx = np.zeros(N)
    fy = np.zeros(N)
    wx = np.zeros(N)
    wy = np.zeros(N)
    mask = np.array([False, False, False])
    density = 1.0
    surface_areas = np.ones(N)
    drag_coeffs = np.ones(N)
    wave_drag = np.ones(N)
    swim_behav = np.zeros(N, dtype=int)
    drags = compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
    assert drags.shape == (N, 2)
    assert np.allclose(drags, 0.0)


def test_assess_fatigue_core_basic():
    sog = np.array([0.5, 1.0])
    heading = np.array([0.0, np.pi/2])
    x_vel = np.zeros(2)
    y_vel = np.zeros(2)
    max_s_U = 0.4
    max_p_U = 1.2
    battery = np.array([1.0, 1.0])
    buf = np.zeros((2, 3))
    swim_speeds, bl_s, prolonged, sprint, sustained = assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, buf)
    assert swim_speeds.shape == (2,)
    assert bl_s.shape == (2,)
    assert isinstance(prolonged[0], (np.bool_, bool))
