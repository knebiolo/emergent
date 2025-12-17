"""Numba-accelerated merged battery kernel."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for merged_battery_numba")

import numpy as np

@njit
def merged_battery(battery, effort, speed, dt):
    N = battery.shape[0]
    out = np.empty(N, dtype=np.float64)
    a = 0.01
    b = 0.005
    for i in range(N):
        cons = effort[i] * (a + b * speed[i] * speed[i]) * dt
        out[i] = battery[i] - cons
    return out
