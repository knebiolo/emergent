"""Physiology helpers: drag computation and fatigue assessment.

This module contains numpy reference implementations ported from legacy
`sockeye.py`. Numba-accelerated variants can be added later behind
an optional dependency flag.
"""
import numpy as np

# Delegate drag computation to the canonical implementation in drags.py
from emergent.fish_passage.drags import compute_drags


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
