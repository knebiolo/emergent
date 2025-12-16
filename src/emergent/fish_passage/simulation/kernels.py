"""Kernel glue to orchestrate numeric kernels for a simulation step.

Provides `compute_step(state, env, dt)` which computes drag vectors,
updates swim speeds buffer, and updates battery using fatigue routines.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

from emergent.fish_passage import drags as _drags
from emergent.fish_passage import fatigue as _fatigue
from emergent.fish_passage import control as _control
from emergent.fish_passage import physiology as _physiology


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

    # --- Motion integration: compute net force and update velocities/positions ---
    # Thrust: can be provided by a PID controller in env or default proportional thrust
    pid: _control.PID_controller = env.get('pid', None)
    if pid is not None:
        # desired heading error (simple placeholder): zero error -> no corrective thrust
        # Here we compute a small upstream error vector if provided in env
        error = env.get('error', np.zeros((state.n_agents, 2)))
        ctrl_out = pid.update(error, dt, state.swim_behav)
        # map control output magnitude to per-agent thrust scalar
        thrust = np.linalg.norm(ctrl_out, axis=1)
    else:
        # default thrust proportional to battery and swim_behav (mode)
        thrust = 0.5 * state.battery

    # Simple mass per agent (based on weight), compute acceleration: a = (thrust_vector - drag) / mass
    mass = np.maximum(state.weight, 0.001)

    # thrust vector aligned with heading
    thrust_x = thrust * np.cos(state.heading)
    thrust_y = thrust * np.sin(state.heading)

    # compute net force (thrust minus drag components)
    net_fx = thrust_x - dr[:, 0]
    net_fy = thrust_y - dr[:, 1]

    # acceleration
    ax = net_fx / mass
    ay = net_fy / mass
    # update velocities
    state.x_vel += ax * dt
    state.y_vel += ay * dt

    # integrate positions (simple Euler)
    state.X += state.x_vel * dt
    state.Y += state.y_vel * dt

    return {'drags': dr, 'swim_speeds': swim_speeds, 'battery': battery_new, 'thrust': thrust}
