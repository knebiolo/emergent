"""Numba-accelerated calc_battery kernel."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for calc_battery_numba")

import numpy as np

@njit
def calc_battery(speeds, efforts, dt):
    N = speeds.shape[0]
    out = np.empty(N, dtype=np.float64)
    a = 0.01
    b = 0.005
    for i in range(N):
        s = speeds[i]
        e = efforts[i]
        out[i] = e * (a + b * s * s) * dt
    return out
