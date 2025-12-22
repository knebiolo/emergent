import math
import numpy as np
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


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
    # clip excessive drags for holding behavior
    drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
    mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
    if np.any(mask_excess):
        scales = 5.0 / drag_mags[mask_excess]
        drags[mask_excess, 0] *= scales
        drags[mask_excess, 1] *= scales
    # apply agent mask
    drags[~mask] = 0.0
    return drags


def compute_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
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


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _compute_drags_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav):
        n = fx.size
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                drags[i, 0] = 0.0
                drags[i, 1] = 0.0
                continue
            rvx = fx[i] - wx[i]
            rvy = fy[i] - wy[i]
            rel = math.hypot(rvx, rvy)
            if rel < 1e-6:
                rel = 1e-6
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = math.hypot(dx, dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return drags


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        # compute swim speeds
        for i in prange(n):
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            swim_speeds[i] = math.hypot(rx, ry)
        # compute bl/s
        bl_s = np.empty(n, dtype=np.float64)
        for i in prange(n):
            bl_s[i] = swim_speeds[i] / (1.0 if 0 else 1.0)
        # mask categories
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        for i in prange(n):
            prolonged[i] = (max_s_U < bl_s[i]) and (bl_s[i] <= max_p_U)
            sprint[i] = bl_s[i] > max_p_U
            sustained[i] = bl_s[i] <= max_s_U
        # write swim speeds into circular buffer last slot
        for i in prange(n):
            swim_speeds_buf[i, -1] = swim_speeds[i]
        return swim_speeds, bl_s, prolonged, sprint, sustained
else:
    def _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
        swim_speeds = np.sqrt((sog * np.cos(heading) - x_vel) ** 2 + (sog * np.sin(heading) - y_vel) ** 2)
        bl_s = swim_speeds / (1.0 if 0 else 1.0)
        prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
        sprint = bl_s > max_p_U
        sustained = bl_s <= max_s_U
        swim_speeds_buf[:, -1] = swim_speeds
        return swim_speeds, bl_s, prolonged, sprint, sustained


if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _merged_swim_drag_fatigue_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        bl_s = np.empty(n, dtype=np.float64)
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        drags = np.zeros((n, 2), dtype=np.float64)
        for i in prange(n):
            if not mask[i]:
                swim_speeds[i] = 0.0
                bl_s[i] = 0.0
                prolonged[i] = False
                sprint[i] = False
                sustained[i] = False
                drags[i, 0] = 0.0
                drags[i, 1] = 0.0
                continue
            # fish velocity components
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            # relative to water
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            s = math.hypot(rx, ry)
            swim_speeds[i] = s
            # preserve existing bl_s semantics (previous implementation used a placeholder)
            bl_s[i] = s
            # fatigue masks
            prolonged[i] = (max_s_U[i] < bl_s[i]) and (bl_s[i] <= max_p_U[i])
            sprint[i] = bl_s[i] > max_p_U[i]
            sustained[i] = bl_s[i] <= max_s_U[i]
            # write into circular buffer last slot
            swim_speeds_buf[i, -1] = swim_speeds[i]

            # compute drag (same as _compute_drags_numba)
            rvx = fx - x_vel[i]
            rvy = fy - y_vel[i]
            rel = math.hypot(rvx, rvy)
            if rel < 1e-6:
                rel = 1e-6
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = math.hypot(dx, dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags
else:
    def _merged_swim_drag_fatigue_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
        fx = sog * np.cos(heading)
        fy = sog * np.sin(heading)
        rx = fx - x_vel
        ry = fy - y_vel
        swim_speeds = np.sqrt(rx * rx + ry * ry)
        bl_s = swim_speeds.copy()
        prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
        sprint = bl_s > max_p_U
        sustained = bl_s <= max_s_U
        swim_speeds_buf[:, -1] = swim_speeds
        rel = np.maximum(np.sqrt((fx - x_vel) ** 2 + (fy - y_vel) ** 2), 1e-6)
        unitx = (fx - x_vel) / rel
        unity = (fy - y_vel) / rel
        relsq = rel * rel
        pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
        dx = pref * unitx
        dy = pref * unity
        drags = np.stack((dx, dy), axis=1)
        mask_arr = np.asarray(mask, dtype=np.bool_)
        # clip excessive drags for holding behavior
        drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
        mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
        if np.any(mask_excess):
            scales = 5.0 / drag_mags[mask_excess]
            drags[mask_excess, 0] *= scales
            drags[mask_excess, 1] *= scales
        drags[~mask_arr] = 0.0
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags


if _HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        for i in prange(dr.shape[0]):
            out[i,0] = dr[i,0]
            out[i,1] = dr[i,1]
        return out
else:
    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numpy(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        out[:,0] = dr[:,0]
        out[:,1] = dr[:,1]
        return out
