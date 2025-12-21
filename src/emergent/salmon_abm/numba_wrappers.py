import numpy as np
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from numba import njit, prange
    import math
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

from .drags import _compute_drags_numba as _compute_drags_numba_from_drags, _compute_drags_numpy as _compute_drags_numpy_from_drags


def get_arr(x):
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


# --- Numba kernels and numpy fallbacks copied verbatim from monolith ---
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


    @njit(cache=True, parallel=True)
    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numba_from_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        # copy into out
        for i in prange(dr.shape[0]):
            out[i,0] = dr[i,0]
            out[i,1] = dr[i,1]
        return out


    @njit(parallel=True, cache=True)
    def _project_points_onto_line_numba(xs_line, ys_line, px, py):
        # compute segment vectors
        S = xs_line.size - 1
        seg_x0 = xs_line[:S]
        seg_y0 = ys_line[:S]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.empty(S, dtype=np.float64)
        for j in range(S):
            seg_len[j] = math.hypot(vx[j], vy[j])
        cumlen = np.empty(S + 1, dtype=np.float64)
        cumlen[0] = 0.0
        for j in range(S):
            cumlen[j + 1] = cumlen[j] + seg_len[j]

        M = px.size
        out = np.empty(M, dtype=np.float64)
        for i in prange(M):
            best_d2 = 1e308
            best_dist = 0.0
            xi = px[i]
            yi = py[i]
            for j in range(S):
                x0 = seg_x0[j]
                y0 = seg_y0[j]
                dx = vx[j]
                dy = vy[j]
                denom = dx * dx + dy * dy
                if denom == 0.0:
                    t = 0.0
                else:
                    t = ((xi - x0) * dx + (yi - y0) * dy) / denom
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                cx = x0 + t * dx
                cy = y0 + t * dy
                d2 = (xi - cx) * (xi - cx) + (yi - cy) * (yi - cy)
                if d2 < best_d2:
                    best_d2 = d2
                    best_dist = cumlen[j] + t * seg_len[j]
            out[i] = best_dist
        return out


    @njit(parallel=True, cache=True)
    def _swim_speeds_numba(x_vel, y_vel, sog, heading):
        n = sog.size
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            out[i] = math.hypot(rx, ry)
        return out


    @njit(parallel=True, cache=True)
    def _swim_speeds_numba_v2(x_vel, y_vel, sog, cos_h, sin_h, out):
        n = sog.size
        for i in prange(n):
            fx = sog[i] * cos_h[i]
            fy = sog[i] * sin_h[i]
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            out[i] = math.hypot(rx, ry)
        return out


    @njit(parallel=True, cache=True)
    def _calc_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        n = battery.size
        # apply sustained recovery
        for i in prange(n):
            if mask_sustained[i]:
                battery[i] = battery[i] + per_rec[i]
        # non-sustained: scale battery by remaining ttf
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
        # clip
        for i in prange(n):
            if battery[i] < 0.0:
                battery[i] = 0.0
            elif battery[i] > 1.0:
                battery[i] = 1.0
        return battery


    @njit(parallel=True, cache=True)
    def _merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
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
            # clip
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0
            battery[i] = b
        return battery


    @njit(parallel=True, cache=True)
    def _bout_distance_numba(prev_X, X, prev_Y, Y):
        n = prev_X.shape[0]
        dist = np.empty(n, dtype=np.float64)
        for i in prange(n):
            dx = prev_X[i] - X[i]
            dy = prev_Y[i] - Y[i]
            dist[i] = math.sqrt(dx * dx + dy * dy)
        return dist


    @njit(parallel=True, cache=True)
    def _time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
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


    @njit(parallel=True, cache=True)
    def _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
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


    @njit(parallel=True, cache=True)
    def _drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery, per_rec, ttf, dt, update_battery):
        n = sog.size
        # outputs
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
            fx = sog[i] * math.cos(heading[i])
            fy = sog[i] * math.sin(heading[i])
            relx = fx - x_vel[i]
            rely = fy - y_vel[i]
            rel = math.hypot(relx, rely)
            if rel < 1e-12:
                rel = 1e-12
            swim_speeds[i] = rel
            bl_s[i] = rel
            unitx = relx / rel
            unity = rely / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            # clip for holding behavior
            if swim_behav[i] == 3:
                mag = math.hypot(dx, dy)
                if mag > 5.0:
                    scale = 5.0 / mag
                    dx *= scale
                    dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
            # battery update if requested
            if update_battery:
                b = battery[i]
                # determine sustained by per_rec>0 as a proxy
                if per_rec is not None and per_rec.size == n and per_rec[i] > 0.0:
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
                # clip
                if b < 0.0:
                    b = 0.0
                elif b > 1.0:
                    b = 1.0
                battery[i] = b
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags, battery

else:
    def _assess_fatigue_core(sog, heading, x_vel, y_vel, max_s_U, max_p_U, battery, swim_speeds_buf):
        swim_speeds = np.sqrt((sog * np.cos(heading) - x_vel) ** 2 + (sog * np.sin(heading) - y_vel) ** 2)
        bl_s = swim_speeds / (1.0 if 0 else 1.0)
        prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
        sprint = bl_s > max_p_U
        sustained = bl_s <= max_s_U
        swim_speeds_buf[:, -1] = swim_speeds
        return swim_speeds, bl_s, prolonged, sprint, sustained

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
        pref = -0.5 * (1.0 * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
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

    def _drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out):
        dr = _compute_drags_numpy_from_drags(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav)
        out[:,0] = dr[:,0]
        out[:,1] = dr[:,1]
        return out

    def _project_points_onto_line_numba(xs_line, ys_line, px, py):
        # fallback to numpy implementation
        seg_x0 = xs_line[:-1]
        seg_y0 = ys_line[:-1]
        seg_x1 = xs_line[1:]
        seg_y1 = ys_line[1:]
        vx = seg_x1 - seg_x0
        vy = seg_y1 - seg_y0
        seg_len = np.hypot(vx, vy)
        cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
        M = px.size
        px_e = px[:, None]
        py_e = py[:, None]
        x0_e = seg_x0[None, :]
        y0_e = seg_y0[None, :]
        vx_e = vx[None, :]
        vy_e = vy[None, :]
        wx = px_e - x0_e
        wy = py_e - y0_e
        denom = vx_e * vx_e + vy_e * vy_e
        denom = np.where(denom == 0, 1e-12, denom)
        t = (wx * vx_e + wy * vy_e) / denom
        t_clamped = np.clip(t, 0.0, 1.0)
        cx = x0_e + t_clamped * vx_e
        cy = y0_e + t_clamped * vy_e
        d2 = (px_e - cx) ** 2 + (py_e - cy) ** 2
        idx = np.argmin(d2, axis=1)
        chosen_t = t_clamped[np.arange(M), idx]
        chosen_seg = idx
        distances_along = cumlen[chosen_seg] + chosen_t * seg_len[chosen_seg]
        return distances_along

    def _swim_speeds_numba(x_vel, y_vel, sog, heading):
        fish_velocities_x = sog * np.cos(heading)
        fish_velocities_y = sog * np.sin(heading)
        relx = fish_velocities_x - x_vel
        rely = fish_velocities_y - y_vel
        return np.sqrt(relx * relx + rely * rely)

    def _swim_speeds_numba_v2(x_vel, y_vel, sog, cos_h, sin_h, out):
        fx = sog * cos_h
        fy = sog * sin_h
        relx = fx - x_vel
        rely = fy - y_vel
        out[:] = np.sqrt(relx * relx + rely * rely)
        return out

    def _calc_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        battery = battery.copy()
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

    def _merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
        battery = battery.copy()
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

    def _bout_distance_numba(prev_X, X, prev_Y, Y):
        dx = prev_X - X
        dy = prev_Y - Y
        return np.sqrt(dx * dx + dy * dy)

    def _time_to_fatigue_numba(swim_speeds, mask_prolonged, mask_sprint, a_p, b_p, a_s, b_s):
        ttf = np.full_like(swim_speeds, np.nan, dtype=float)
        ttf = np.where(mask_prolonged, np.exp(a_p + swim_speeds * b_p), ttf)
        ttf = np.where(mask_sprint, np.exp(a_s + swim_speeds * b_s), ttf)
        return ttf

    def _swim_core_numba(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
        vx = fv0x + accx * dt
        vy = fv0y + accy * dt
        vx = np.where(~tired_mask, vx + pidx, vx)
        vy = np.where(~tired_mask, vy + pidy, vy)
        vx = np.where((~mask) | dead_mask, 0.0, vx)
        vy = np.where((~mask) | dead_mask, 0.0, vy)
        return np.stack((vx * dt, vy * dt), axis=1)

    def _drag_and_battery_numba(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, battery, per_rec, ttf, dt, update_battery):
        n = sog.size
        swim_speeds = np.empty(n, dtype=np.float64)
        bl_s = np.empty(n, dtype=np.float64)
        prolonged = np.empty(n, dtype=np.bool_)
        sprint = np.empty(n, dtype=np.bool_)
        sustained = np.empty(n, dtype=np.bool_)
        fx = sog * np.cos(heading)
        fy = sog * np.sin(heading)
        relx = fx - x_vel
        rely = fy - y_vel
        rel = np.sqrt(relx * relx + rely * rely)
        rel = np.where(rel == 0, 1e-12, rel)
        swim_speeds[:] = rel
        bl_s[:] = rel
        unitx = relx / rel
        unity = rely / rel
        relsq = rel * rel
        pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
        dx = pref * unitx
        dy = pref * unity
        drags = np.stack((dx, dy), axis=1)
        mask_arr = np.asarray(mask, dtype=np.bool_)
        drag_mags = np.sqrt(drags[:,0]**2 + drags[:,1]**2)
        mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
        if np.any(mask_excess):
            scales = 5.0 / drag_mags[mask_excess]
            drags[mask_excess,0] *= scales
            drags[mask_excess,1] *= scales
        drags[~mask_arr] = 0.0
        batt = battery.copy()
        if update_battery:
            for i in range(batt.size):
                if per_rec is not None and per_rec.size == batt.size and per_rec[i] > 0.0:
                    batt[i] = batt[i] + per_rec[i]
                else:
                    t0 = ttf[i] * batt[i]
                    if t0 <= 0.0:
                        batt[i] = 0.0
                    else:
                        t1 = t0 - dt
                        ratio = t1 / t0
                        if ratio < 0.0:
                            ratio = 0.0
                        batt[i] = batt[i] * ratio
        np.clip(batt, 0.0, 1.0, out=batt)
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags, batt


# --- Safe wrapper helpers (ensure contiguous arrays for numba calls) ---
def _wrap_project_points_onto_line_numba(xs_line, ys_line, px, py):
    xs = np.ascontiguousarray(xs_line, dtype=np.float64)
    ys = np.ascontiguousarray(ys_line, dtype=np.float64)
    pxx = np.ascontiguousarray(px, dtype=np.float64)
    pyy = np.ascontiguousarray(py, dtype=np.float64)
    return _project_points_onto_line_numba(xs, ys, pxx, pyy)


def _wrap_drag_fun_numba(fx, fy, wx, wy, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, out=None):
    fx_a = np.ascontiguousarray(fx, dtype=np.float64)
    fy_a = np.ascontiguousarray(fy, dtype=np.float64)
    wx_a = np.ascontiguousarray(wx, dtype=np.float64)
    wy_a = np.ascontiguousarray(wy, dtype=np.float64)
    mask_a = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_), dtype=np.bool_)
    sa = np.ascontiguousarray(surface_areas, dtype=np.float64)
    dc = np.ascontiguousarray(drag_coeffs, dtype=np.float64)
    wd = np.ascontiguousarray(wave_drag, dtype=np.float64)
    if out is None:
        out = np.zeros((fx_a.size, 2), dtype=np.float64)
    else:
        out = np.ascontiguousarray(out, dtype=np.float64)
    return _drag_fun_numba(fx_a, fy_a, wx_a, wy_a, mask_a, float(density), sa, dc, wd, np.ascontiguousarray(np.asarray(swim_behav, dtype=np.int64)), out)


def _wrap_merged_battery_numba(battery, per_rec, ttf, mask_sustained, dt):
    batt = np.ascontiguousarray(battery, dtype=np.float64)
    perr = np.ascontiguousarray(per_rec, dtype=np.float64)
    ttf_a = np.ascontiguousarray(ttf, dtype=np.float64)
    mask_a = np.ascontiguousarray(np.asarray(mask_sustained, dtype=np.bool_), dtype=np.bool_)
    return _merged_battery_numba(batt, perr, ttf_a, mask_a, float(dt))


# Warmup helpers to precompile numba functions if available
def _numba_warmup(m=None):
    try:
        if not _HAS_NUMBA:
            return
        if m is None:
            m = max(64, 8)
        else:
            m = int(m)
        d = np.zeros(m, dtype=np.float64)
        b = np.ones(m, dtype=np.bool_)
        bi = np.zeros(m, dtype=np.int64)
        _ = _compute_drags_numba_from_drags(d, d, d, d, b, 1.0, np.ones(m, dtype=np.float64), np.ones(m, dtype=np.float64), np.ones(m, dtype=np.float64), bi)
        _ = _bout_distance_numba(d, d, d, d)
        _ = _time_to_fatigue_numba(d, b, np.zeros(m, dtype=np.bool_), 0.0, 0.0, 0.0, 0.0)
        _ = _project_points_onto_line_numba(d, d, d, d)
        _ = _swim_speeds_numba(d, d, d, d)
        _ = _calc_battery_numba(d, d, d, b, 0.1)
    except Exception:
        pass


def _numba_warmup_for_sim(sim):
    try:
        if not _HAS_NUMBA:
            return
        n = max(1024, int(getattr(sim, 'num_agents', 128)))
        _numba_warmup(m=n)
        na = int(getattr(sim, 'num_agents', n))
        max_ts = int(getattr(sim, 'swim_speeds', np.zeros((na,1))).shape[1])
        ones = np.ones(na, dtype=np.float64)
        zeros = np.zeros(na, dtype=np.float64)
        bmask = np.ones(na, dtype=np.bool_)
        bi = np.zeros(na, dtype=np.int64)
        try:
            _compute_drags_numba_from_drags(ones, ones, ones, ones, bmask, 1.0, ones, ones, ones, bi)
        except Exception:
            pass
        try:
            _swim_speeds_numba(ones, ones, ones, ones)
        except Exception:
            pass
        try:
            buf = np.zeros((na, max_ts), dtype=np.float64)
            _assess_fatigue_core(ones, ones, ones, ones, ones, ones, ones, buf)
        except Exception:
            pass
    except Exception:
        pass

