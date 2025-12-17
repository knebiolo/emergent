"""Numba-accelerated swim core integrator."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for swim_core_numba")

import numpy as np

@njit
def swim_core(positions, headings, speeds, env_forces, dt):
    # positions: (N,2)
    N = positions.shape[0]
    new_positions = np.empty_like(positions)
    new_speeds = np.empty_like(speeds)
    for i in range(N):
        x = positions[i, 0]
        y = positions[i, 1]
        h = headings[i]
        s = speeds[i]
        dx = np.cos(h) * s * dt
        dy = np.sin(h) * s * dt
        if env_forces is not None:
            dx += env_forces[i, 0] * dt
            dy += env_forces[i, 1] * dt
        new_positions[i, 0] = x + dx
        new_positions[i, 1] = y + dy
        new_speeds[i] = s
    return new_positions, new_speeds
