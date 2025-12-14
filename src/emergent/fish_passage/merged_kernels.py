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
    def merged_swim_drag_fatigue(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
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
            fx = sog[i] * np.cos(heading[i])
            fy = sog[i] * np.sin(heading[i])
            rx = fx - x_vel[i]
            ry = fy - y_vel[i]
            s = np.hypot(rx, ry)
            swim_speeds[i] = s
            bl_s[i] = s
            prolonged[i] = (max_s_U[i] < bl_s[i]) and (bl_s[i] <= max_p_U[i])
            sprint[i] = bl_s[i] > max_p_U[i]
            sustained[i] = bl_s[i] <= max_s_U[i]
            swim_speeds_buf[i, -1] = swim_speeds[i]

            rvx = fx - x_vel[i]
            rvy = fy - y_vel[i]
            rel = np.hypot(rvx, rvy)
            if rel < 1e-6:
                rel = 1e-6
            unitx = rvx / rel
            unity = rvy / rel
            relsq = rel * rel
            pref = -0.5 * (density * 1000.0) * (surface_areas[i] / (100.0 ** 2)) * drag_coeffs[i] * relsq * wave_drag[i]
            dx = pref * unitx
            dy = pref * unity
            mag = np.hypot(dx, dy)
            if swim_behav[i] == 3 and mag > 5.0:
                scale = 5.0 / mag
                dx *= scale
                dy *= scale
            drags[i, 0] = dx
            drags[i, 1] = dy
        return swim_speeds, bl_s, prolonged, sprint, sustained, drags


def merged_swim_drag_fatigue(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf):
    """Numpy fallback implementing merged swim/drag/fatigue logic.

    Returns: swim_speeds, bl_s, prolonged, sprint, sustained, drags
    """
    fx = sog * np.cos(heading)
    fy = sog * np.sin(heading)
    rx = fx - x_vel
    ry = fy - y_vel
    swim_speeds = np.sqrt(rx * rx + ry * ry)
    # Respect mask semantics: zero-out entries where mask is False
    mask_arr = np.asarray(mask, dtype=bool)
    swim_speeds_masked = swim_speeds.copy()
    swim_speeds_masked[~mask_arr] = 0.0
    bl_s = swim_speeds_masked.copy()
    prolonged = (max_s_U < bl_s) & (bl_s <= max_p_U)
    sprint = bl_s > max_p_U
    sustained = bl_s <= max_s_U
    # ensure inactive agents are False (legacy numba sets False when not mask)
    prolonged[~mask_arr] = False
    sprint[~mask_arr] = False
    sustained[~mask_arr] = False
    # write only for active agents (mirror legacy behavior)
    swim_speeds_buf[mask_arr, -1] = swim_speeds[mask_arr]
    rel = np.maximum(np.sqrt((fx - x_vel) ** 2 + (fy - y_vel) ** 2), 1e-6)
    unitx = (fx - x_vel) / rel
    unity = (fy - y_vel) / rel
    relsq = rel * rel
    pref = -0.5 * (density * 1000.0) * (surface_areas / (100.0 ** 2)) * drag_coeffs * relsq * wave_drag
    dx = pref * unitx
    dy = pref * unity
    drags = np.stack((dx, dy), axis=1)
    # clip excessive drags for holding behavior
    drag_mags = np.sqrt(drags[:, 0] ** 2 + drags[:, 1] ** 2)
    mask_excess = (swim_behav == 3) & (drag_mags > 5.0)
    if np.any(mask_excess):
        scales = 5.0 / drag_mags[mask_excess]
        drags[mask_excess, 0] *= scales
        drags[mask_excess, 1] *= scales
    # zero drags for inactive agents
    drags[~mask_arr] = 0.0
    return swim_speeds_masked, bl_s, prolonged, sprint, sustained, drags


def wrap_merged_battery(battery, per_rec, ttf, mask_sustained, dt):
    """Compatibility shim matching sockeye's `_wrap_merged_battery_numba` naming.

    Delegates to `fatigue.merged_battery` to keep legacy call sites working.
    """
    from emergent.fish_passage import fatigue
    return fatigue.merged_battery(battery, per_rec, ttf, mask_sustained, dt)


def drag_and_battery(sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, per_rec, ttf, dt, update_battery=True, swim_speeds_buf=None):
    """Public wrapper that computes swim speeds, drags and optionally updates battery.

    This composes `merged_swim_drag_fatigue` and `fatigue.merged_battery` to provide
    the single-pass API used by some legacy code paths.

    Returns: (swim_speeds, bl_s, prolonged, sprint, sustained, drags, battery)
    """
    from emergent.fish_passage import fatigue

    # Use the pure fish_passage implementation (no delegation to legacy code).
    n = sog.size
    if swim_speeds_buf is None:
        swim_speeds_buf = np.zeros((n, 4), dtype=np.float64)

    ss, bl_s, prolonged, sprint, sustained, drags = merged_swim_drag_fatigue(
        sog, heading, x_vel, y_vel, mask, density, surface_areas, drag_coeffs, wave_drag, swim_behav, max_s_U, max_p_U, battery, swim_speeds_buf
    )

    if update_battery:
        # To ensure bit-for-bit parity with legacy `_drag_and_battery_numba`,
        # perform the battery update in an explicit per-element loop using
        # the same ordering of operations and branching.
        per_rec_arr = np.asarray(per_rec) if per_rec is not None else np.zeros_like(battery)
        n = battery.size
        new_batt = battery.copy().astype(np.float64)
        for i in range(n):
            # Legacy loop skips inactive agents entirely
            if not bool(mask[i]):
                continue
            b = float(new_batt[i])
            if per_rec_arr.size == n and per_rec_arr[i] > 0.0:
                b = b + float(per_rec_arr[i])
            else:
                t0 = float(ttf[i]) * b
                if t0 <= 0.0:
                    b = 0.0
                else:
                    t1 = t0 - float(dt)
                    ratio = t1 / t0
                    if ratio < 0.0:
                        ratio = 0.0
                    b = b * ratio
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0
            new_batt[i] = b
    else:
        new_batt = battery.copy()

    # Legacy `_drag_and_battery_numba` returned False for prolonged/sprint/sustained
    # (caller computed thresholds externally). Ensure we match that interface.
    n = sog.size
    prolonged = np.zeros(n, dtype=bool)
    sprint = np.zeros(n, dtype=bool)
    sustained = np.zeros(n, dtype=bool)

    return ss, bl_s, prolonged, sprint, sustained, drags, new_batt
