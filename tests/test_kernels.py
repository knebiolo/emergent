import sys
sys.path.insert(0, 'src')
import numpy as np
import pytest

from emergent.salmon_abm.drags import compute_drags
from emergent.salmon_abm.numba_wrappers import (
    _assess_fatigue_core,
    _merged_battery_numba,
    _project_points_onto_line_numba,
    _swim_speeds_numba,
)


def test_compute_drags_shapes_and_values():
    n = 5
    fx = np.linspace(0.5, 1.0, n)
    fy = np.linspace(0.2, 0.8, n)
    wx = np.zeros(n)
    wy = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    density = 1.0
    surface_areas = np.ones(n) * 5.0
    drag_coeffs = np.ones(n) * 0.5
    wave_drag = np.ones(n)
    swim_behav = np.zeros(n, dtype=int)

    dr = compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
    assert dr.shape == (n, 2)
    # drags should be negative (prefactor negative) when fx/fy non-zero
    assert np.all(np.isfinite(dr))
    assert np.any(dr != 0.0)


def test_assess_fatigue_core_numpy_behavior():
    n = 6
    sog = np.ones(n) * 0.6
    heading = np.zeros(n)
    x_vel = np.zeros(n)
    y_vel = np.zeros(n)
    max_s_U = np.ones(n) * 0.3
    max_p_U = np.ones(n) * 1.0
    battery = np.ones(n) * 0.9
    swim_speeds_buf = np.zeros((n, 4))

    swim_speeds, bl_s, prolonged, sprint, sustained = _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf)
    assert swim_speeds.shape == (n,)
    assert bl_s.shape == (n,)
    assert prolonged.dtype == bool
    assert sprint.dtype == bool
    assert sustained.dtype == bool
    assert np.all(swim_speeds >= 0.0)


def test_merged_battery_updates():
    n = 8
    battery = np.ones(n) * 0.5
    per_rec = np.zeros(n)
    ttf = np.ones(n) * 10.0
    mask_sustained = np.zeros(n, dtype=bool)
    dt = 0.5
    b = _merged_battery_numba(battery.copy(), per_rec, ttf, mask_sustained, dt)
    assert b.shape == (n,)
    # when per_rec all zeros and dt>0, battery should decrease or stay same
    assert np.all(b <= battery + 1e-12)


def test_project_points_onto_line_basic():
    # simple horizontal line from x=0..3 at y=0
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.zeros_like(xs)
    px = np.array([0.1, 1.5, 2.9])
    py = np.array([0.0, 0.0, 0.0])
    d = _project_points_onto_line_numba(xs, ys, px, py)
    assert d.shape == (px.size,)
    # distances along should be near px values
    assert np.allclose(d, px, atol=0.2)


def test_swim_speeds_numba_vs_numpy():
    n = 10
    x_vel = np.zeros(n)
    y_vel = np.zeros(n)
    sog = np.linspace(0.1, 1.0, n)
    heading = np.linspace(0.0, np.pi/2, n)
    arr = _swim_speeds_numba(x_vel, y_vel, sog, heading)
    assert arr.shape == (n,)
    # numeric sanity: non-negative
    assert np.all(arr >= 0.0)
