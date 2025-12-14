"""Fatigue and battery update kernels ported from legacy sockeye._calc_battery_numba.

Provides `calc_battery` which matches the legacy behaviour with a numpy
implementation and an optional numba-warmed variant when available.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def calc_battery(battery: Sequence[float], per_rec: Sequence[float], ttf: Sequence[float], mask_sustained: Sequence[bool], dt: float) -> np.ndarray:
    """Numpy fallback for `_calc_battery_numba`.

    Returns updated battery array (values clipped to [0,1]).
    """
    battery = np.asarray(battery, dtype=np.float64).copy()
    per_rec = np.asarray(per_rec, dtype=np.float64)
    ttf = np.asarray(ttf, dtype=np.float64)
    mask_sustained = np.asarray(mask_sustained, dtype=np.bool_)

    battery[mask_sustained] += per_rec[mask_sustained]
    mask_non = ~mask_sustained
    ttf0 = ttf[mask_non] * battery[mask_non]
    ttf1 = ttf0 - dt
    safe = ttf0 != 0
    ratio = np.ones_like(ttf0)
    ratio[safe] = np.maximum(0.0, ttf1[safe] / ttf0[safe])
    battery[mask_non] = battery[mask_non] * ratio
    np.clip(battery, 0.0, 1.0, out=battery)
    return battery


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def calc_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        n = battery.size
        for i in prange(n):
            if mask_sustained[i]:
                battery[i] = battery[i] + per_rec[i]
        for i in prange(n):
            if not mask_sustained[i]:
                ttf0 = ttf[i] * battery[i]
                if ttf0 <= 0.0:
                    battery[i] = 0.0
                else:
                    ttf1 = ttf0 - dt
                    ratio = ttf1 / ttf0
                    if ratio < 0.0:
                        ratio = 0.0
                    battery[i] = battery[i] * ratio
        for i in prange(n):
            if battery[i] < 0.0:
                battery[i] = 0.0
            elif battery[i] > 1.0:
                battery[i] = 1.0
        return battery
