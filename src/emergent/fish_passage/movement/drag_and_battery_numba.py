"""Numba-accelerated drag and battery kernel."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for drag_and_battery_numba")

import numpy as np

@njit
def drag_and_battery(positions, speeds, headings, env_fields, dt):
    N = speeds.shape[0]
    forces = np.zeros((N, 2), dtype=np.float64)
    battery_updates = np.zeros(N, dtype=np.float64)
    k = 0.1
    for i in range(N):
        s = speeds[i]
        h = headings[i]
        drag_mag = k * s * s
        forces[i, 0] = -np.cos(h) * drag_mag
        forces[i, 1] = -np.sin(h) * drag_mag
        battery_updates[i] = s * drag_mag * dt
    return forces, battery_updates
