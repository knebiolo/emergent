"""Numba-accelerated time-to-fatigue kernel."""
try:
    from numba import njit
except Exception:
    raise ImportError("numba is required for time_to_fatigue_numba")

import numpy as np

@njit
def time_to_fatigue(energy_states, rate):
    N = energy_states.shape[0]
    out = np.empty(N, dtype=np.float64)
    # assume rate > 0 passed in
    for i in range(N):
        r = rate
        if r <= 0:
            r = 1e-8
        out[i] = energy_states[i] / r
    return out
