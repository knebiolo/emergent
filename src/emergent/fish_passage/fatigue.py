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
    ratio = np.zeros_like(ttf0)
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


def merged_battery(battery: Sequence[float], per_rec: Sequence[float], ttf: Sequence[float], mask_sustained: Sequence[bool], dt: float) -> np.ndarray:
    """Single-pass merged battery update (numpy fallback of `_merged_battery_numba`)."""
    battery = np.asarray(battery, dtype=np.float64).copy()
    per_rec = np.asarray(per_rec, dtype=np.float64)
    ttf = np.asarray(ttf, dtype=np.float64)
    mask_sustained = np.asarray(mask_sustained, dtype=np.bool_)

    for i in range(battery.size):
        if mask_sustained[i]:
            battery[i] = battery[i] + per_rec[i]
        else:
            t0 = ttf[i] * battery[i]
            if t0 <= 0.0:
                battery[i] = 0.0
            else:
                t1 = t0 - dt
                ratio = t1 / t0
                if ratio < 0.0:
                    ratio = 0.0
                battery[i] = battery[i] * ratio
    np.clip(battery, 0.0, 1.0, out=battery)
    return battery

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        n = battery.size
        for i in prange(n):
            b = battery[i]
            if mask_sustained[i]:
                b = b + per_rec[i]
            else:
                t0 = ttf[i] * b
                if t0 <= 0.0:
                    b = 0.0
                else:
                    t1 = t0 - dt
                    ratio = t1 / t0
                    if ratio < 0.0:
                        ratio = 0.0
                    b = b * ratio
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0
            battery[i] = b
        return battery


    def time_to_fatigue(swim_speeds: Sequence[float], mask_prolonged: Sequence[bool], mask_sprint: Sequence[bool], a_p: float, b_p: float, a_s: float, b_s: float) -> np.ndarray:
        """Numpy fallback for `_time_to_fatigue_numba`.

        Computes `ttf` using the legacy logic: ttf = exp(a + s*b) where the
        prolonged mask uses (a_p, b_p) and the sprint mask uses (a_s, b_s).
        If both masks are False, result is NaN for that index.
        """
        swim_speeds = np.asarray(swim_speeds, dtype=np.float64)
        mask_prolonged = np.asarray(mask_prolonged, dtype=np.bool_)
        mask_sprint = np.asarray(mask_sprint, dtype=np.bool_)

        ttf = np.full_like(swim_speeds, np.nan, dtype=np.float64)
        if np.any(mask_prolonged):
            ttf = np.where(mask_prolonged, np.exp(a_p + swim_speeds * b_p), ttf)
        if np.any(mask_sprint):
            ttf = np.where(mask_sprint, np.exp(a_s + swim_speeds * b_s), ttf)
        return ttf

    if _HAS_NUMBA:
        @njit(parallel=True, cache=True)
        def time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
            n = swim_speeds.shape[0]
            ttf = np.empty(n, dtype=np.float64)
            for i in prange(n):
                ttf[i] = np.nan
                s = swim_speeds[i]
                if mask_prolonged[i]:
                    ttf[i] = math.exp(a_p + s * b_p)
                if mask_sprint[i]:
                    ttf[i] = math.exp(a_s + s * b_s)
            return ttf


    def bout_distance(prev_X: Sequence[float], X: Sequence[float], prev_Y: Sequence[float], Y: Sequence[float]) -> np.ndarray:
        """Vectorized distance between previous and current positions per agent."""
        prev_X = np.asarray(prev_X, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        prev_Y = np.asarray(prev_Y, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        return np.sqrt((prev_X - X) ** 2 + (prev_Y - Y) ** 2)

    if _HAS_NUMBA:
        @njit(parallel=True, cache=True)
        def bout_distance_numba(prev_X, X, prev_Y, Y):
            n = prev_X.shape[0]
            dist = np.empty(n, dtype=np.float64)
            for i in prange(n):
                dx = prev_X[i] - X[i]
                dy = prev_Y[i] - Y[i]
                dist[i] = math.sqrt(dx * dx + dy * dy)
            return dist
