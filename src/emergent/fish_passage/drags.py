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
    def _compute_drags_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
        n = fx.shape[0]
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                continue
            rvx = fx[i] - wx[i]
            rvy = fy[i] - wy[i]
            rel = rvx * rvx + rvy * rvy
            if rel < 1e-12:
                rel = 1e-12
            rel = np.sqrt(rel)
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = np.sqrt(dx * dx + dy * dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return drags


def _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    relative_velocities_x = fx - wx
    relative_velocities_y = fy - wy
    rel_norms = np.sqrt(relative_velocities_x ** 2 + relative_velocities_y ** 2)
    rel_norms_safe = np.maximum(rel_norms, 1e-6)
    unit_x = relative_velocities_x / rel_norms_safe
    unit_y = relative_velocities_y / rel_norms_safe
    relsq = rel_norms ** 2
    pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
    dx = pref * unit_x
    dy = pref * unit_y
    drags = np.stack((dx, dy), axis=1)
    drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
    mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
    if np.any(mask_excess):
        scales = 5.0 / drag_mags[mask_excess]
        drags[mask_excess, 0] *= scales
        drags[mask_excess, 1] *= scales
    drags[~mask] = 0.0
    return drags


def compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
    """Compute drag vectors for agents.

    Parameters mirror the original implementation in `sockeye.py`.
    Returns an (N,2) float64 array of drag vectors.
    """
    if _HAS_NUMBA:
        fx = np.ascontiguousarray(fx, dtype=np.float64)
        fy = np.ascontiguousarray(fy, dtype=np.float64)
        wx = np.ascontiguousarray(wx, dtype=np.float64)
        wy = np.ascontiguousarray(wy, dtype=np.float64)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        surface_areas = np.ascontiguousarray(surface_areas, dtype=np.float64)
        drag_coeffs = np.ascontiguousarray(drag_coeffs, dtype=np.float64)
        wave_drag = np.ascontiguousarray(wave_drag, dtype=np.float64)
        swim_behav = np.ascontiguousarray(swim_behav, dtype=np.int64)
        return _compute_drags_numba(fx, fy, wx, wy, mask, float(density), surface_areas, drag_coeffs, wave_drag, swim_behav)
    else:
        return _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
