"""Numba-accelerated bout distance kernel."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for bout_distance_numba")

import numpy as np

@njit
def bout_distance(speeds, mean_duration):
    N = speeds.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        out[i] = speeds[i] * mean_duration
    return out
