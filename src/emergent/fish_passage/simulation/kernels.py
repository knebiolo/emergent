"""Kernel glue to orchestrate numeric kernels for a simulation step.

Provides `compute_step(state, env, dt)` which computes drag vectors,
updates swim speeds buffer, and updates battery using fatigue routines.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from emergent.fish_passage import drags as _drags
from emergent.fish_passage import fatigue as _fatigue


def compute_step(state, env: Dict[str, Any], dt: float) -> Dict[str, Any]:
    """Compute numeric kernels for one timestep.

    Parameters
    - state: SimulationState
    - env: dict with keys `wx`, `wy`, `density` (scalars or per-agent), `mask` (optional)
    - dt: timestep seconds

    Returns dict with keys: `drags`, `swim_speeds`, `battery`
    """
    n = state.n_agents

    # Build fish-over-ground components
    fx = state.sog * np.cos(state.heading)
    fy = state.sog * np.sin(state.heading)

    # environmental water velocities (per-agent arrays expected)
    wx = np.asarray(env.get('wx', np.zeros(n)), dtype=np.float64)
    wy = np.asarray(env.get('wy', np.zeros(n)), dtype=np.float64)

    # agent mask: alive and not already dead
    mask = np.asarray(env.get('mask', (state.dead == 0)), dtype=bool)

    density = float(env.get('density', 1.0))

    # call drag kernel
    dr = _drags.compute_drags(fx, fy, wx, wy, mask, density, state.surface_area, state.drag_coeffs, state.wave_drag, state.swim_behav)

    # compute swim speeds relative to water (used for fatigue)
    swim_speeds = np.sqrt((fx - wx) ** 2 + (fy - wy) ** 2)

    # update circular buffer last column
    if state.swim_speeds_buf.shape[1] >= 1:
        # roll left and set last to new speeds
        state.swim_speeds_buf[:, :-1] = state.swim_speeds_buf[:, 1:]
        state.swim_speeds_buf[:, -1] = swim_speeds

    # determine fatigue masks: prolonged if <= ucrit, sprint if > ucrit
    mask_prolonged = swim_speeds <= state.ucrit
    mask_sprint = swim_speeds > state.ucrit

    # compute time-to-fatigue using default coefficients (tunable)
    a_p, b_p = -1.0, 0.1
    a_s, b_s = -2.0, 0.2
    ttf = _fatigue.time_to_fatigue(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s)

    # simple per-rec recovery: small positive increment when prolonged
    per_rec = np.where(mask_prolonged, 0.01, 0.0)
    mask_sustained = mask_prolonged

    battery_new = _fatigue.merged_battery(state.battery, per_rec, ttf, mask_sustained, dt)
    state.battery[:] = battery_new

    return {'drags': dr, 'swim_speeds': swim_speeds, 'battery': battery_new}
