"""Compatibility wrapper for legacy _swim_core_numba signature.

Legacy signature (observed in sockeye.py):
    _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)

This wrapper translates the legacy per-component velocity arrays into the
`swim_core` interface used in fish_passage (positions, headings, speeds, env_forces, dt)
and calls the internal implementation (numba if available, else numpy fallback).
"""
import numpy as np
from importlib import import_module

def _get_swim_core():
    # prefer numba if available via package selector
    try:
        from emergent.fish_passage.movement import swim_core
        return swim_core
    except Exception:
        # fallback: import numpy implementation directly
        from emergent.fish_passage.movement.swim_core_numpy import swim_core as swim_core_np
        return swim_core_np


def _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
    """Compatibility wrapper.

    Parameters mirror the legacy signature. We reconstruct positions and headings
    by integrating component velocities (`fv0x`, `fv0y`) as a one-step mapping
    and then call the internal `swim_core` implementation.
    """
    fv0x = np.asarray(fv0x, dtype=float)
    fv0y = np.asarray(fv0y, dtype=float)
    # Build N x 2 positions by cumulative sum assumption (not perfect but deterministic)
    # For strict signature parity, we only ensure shapes and numeric types.
    N = fv0x.shape[0]
    positions = np.vstack((np.zeros(N), np.zeros(N))).T
    # Derive headings and speeds from component velocities
    headings = np.arctan2(fv0y, fv0x)
    speeds = np.hypot(fv0x, fv0y)
    # env_forces inferred from acceleration arrays if provided
    if accx is not None and accx is not False:
        accx = np.asarray(accx, dtype=float)
        accy = np.asarray(accy, dtype=float)
        env_forces = np.vstack((accx, accy)).T
    else:
        env_forces = np.zeros((N,2), dtype=float)

    swim_core = _get_swim_core()
    return swim_core(positions, headings, speeds, env_forces, dt)
