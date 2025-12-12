"""Physiology helpers: drag computation and fatigue assessment.

This module contains numpy reference implementations ported from legacy
`sockeye.py`. Numba-accelerated variants can be added later behind
an optional dependency flag.
"""
import numpy as np


def _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    """Vectorized numpy implementation of drag computation.

    Parameters mirror legacy callers. All inputs are numpy arrays of length N.
    Returns an array (N,2) of drag forces (dx, dy).
    """
    fx = np.asarray(fx, dtype=float)
    fy = np.asarray(fy, dtype=float)
    wx = np.asarray(wx, dtype=float)
    wy = np.asarray(wy, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    surface_areas = np.asarray(surface_areas, dtype=float)
    drag_coeffs = np.asarray(drag_coeffs, dtype=float)
    wave_drag = np.asarray(wave_drag, dtype=float)
    swim_behav = np.asarray(swim_behav)

    relative_velocities_x = fx - wx
    relative_velocities_y = fy - wy
    rel_norms = np.sqrt(relative_velocities_x ** 2 + relative_velocities_y ** 2)
    rel_norms_safe = np.maximum(rel_norms, 1e-6)
    unit_x = relative_velocities_x / rel_norms_safe
    unit_y = relative_velocities_y / rel_norms_safe
    relsq = rel_norms ** 2
    # density expected in kg/L or similar in legacy; preserve formula structure
    pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
    dx = pref * unit_x
    dy = pref * unit_y
    drags = np.stack((dx, dy), axis=1)
    # clip excessive drags for holding behavior (swim_behav == 3)
    drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
    mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
    if np.any(mask_excess):
        scales = 5.0 / drag_mags[mask_excess]
        drags[mask_excess, 0] *= scales
        drags[mask_excess, 1] *= scales
    # apply agent mask
    drags[~mask] = 0.0
    return drags


def compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    """Public wrapper: currently uses numpy implementation.

    (A Numba-accelerated backend may be selected in future.)
    """
    return _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)


def assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
    """Numpy implementation of fatigue assessment core.

    Returns (swim_speeds, bl_s, prolonged, sprint, sustained) analogous to legacy code.
    """
    sog = np.asarray(sog, dtype=float)
    heading = np.asarray(heading, dtype=float)
    x_vel = np.asarray(x_vel, dtype=float)
    y_vel = np.asarray(y_vel, dtype=float)
    swim_speeds = np.sqrt((sog * np.cos(heading) - x_vel) ** 2 + (sog * np.sin(heading) - y_vel) ** 2)
    bl_s = swim_speeds / 1.0
    prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
    sprint = bl_s > max_p_U
    sustained = bl_s <= max_s_U
    # write swim speeds into last column of circular buffer
    try:
        swim_speeds_buf[:, -1] = swim_speeds
    except Exception:
        pass
    return swim_speeds, bl_s, prolonged, sprint, sustained
"""
physiology.py

Preamble/Module plan for physiological scaling and swim modes (moved to fish_passage).

Responsibilities (planned):
- Scale behavioral parameters by body length, model swim modes (burst, sustained), and fatigue thresholds.
- Functions:
  - `scale_by_body_length(value, body_length)`
  - `mode_switch(agent, conditions)`
  - `compute_fatigue_thresholds(agent)`

Notes:
- Keep functions small and document assumptions; suitable for unit testing.
"""
