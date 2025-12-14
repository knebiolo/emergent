"""Swim core kernel ported from legacy sockeye._swim_core_numba.

Provides `swim_core` that computes per-agent displacement dx,dy
from velocity, acceleration, PID adjustments, and activity masks.
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


def swim_core(fv0x: Sequence[float], fv0y: Sequence[float], accx: Sequence[float], accy: Sequence[float], pidx: Sequence[float], pidy: Sequence[float], tired_mask: Sequence[bool], dead_mask: Sequence[bool], mask: Sequence[bool], dt: float) -> np.ndarray:
    fv0x = np.asarray(fv0x, dtype=np.float64)
    fv0y = np.asarray(fv0y, dtype=np.float64)
    accx = np.asarray(accx, dtype=np.float64)
    accy = np.asarray(accy, dtype=np.float64)
    pidx = np.asarray(pidx, dtype=np.float64)
    pidy = np.asarray(pidy, dtype=np.float64)
    tired_mask = np.asarray(tired_mask, dtype=np.bool_)
    dead_mask = np.asarray(dead_mask, dtype=np.bool_)
    mask = np.asarray(mask, dtype=np.bool_)

    vx = fv0x + accx * dt
    vy = fv0y + accy * dt
    vx = np.where(~tired_mask, vx + pidx, vx)
    vy = np.where(~tired_mask, vy + pidy, vy)
    vx = np.where((~mask) | dead_mask, 0.0, vx)
    vy = np.where((~mask) | dead_mask, 0.0, vy)
    return np.stack((vx * dt, vy * dt), axis=1)


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
        n = fv0x.shape[0]
        dxdy = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i] or dead_mask[i]:
                continue
            vx = fv0x[i] + accx[i] * dt
            vy = fv0y[i] + accy[i] * dt
            if not tired_mask[i]:
                vx += pidx[i]
                vy += pidy[i]
            dxdy[i, 0] = vx * dt
            dxdy[i, 1] = vy * dt
        return dxdy
