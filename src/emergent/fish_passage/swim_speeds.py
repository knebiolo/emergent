import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def swim_speeds(x_vel, y_vel, sog, heading):
        n = sog.size
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            fx = sog[i] * np.cos(heading[i])
            fy = sog[i] * np.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            out[i] = np.hypot(rx, ry)
        return out
else:
    def swim_speeds(x_vel, y_vel, sog, heading):
        fish_velocities_x = sog * np.cos(heading)
        fish_velocities_y = sog * np.sin(heading)
        relx = fish_velocities_x - x_vel
        rely = fish_velocities_y - y_vel
        return np.sqrt(relx * relx + rely * rely)
