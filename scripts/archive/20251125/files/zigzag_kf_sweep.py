"""Zig-zag waypoint Kf sweep

Creates a zig-zag waypoint pattern and runs headless simulations sweeping Kf
around a baseline autopilot (Kp/Ki/Kd). Allows overriding max rudder for tests.
Writes per-run trajectory CSVs and a summary CSV in the CWD.
"""
import os
import csv
import math
import numpy as np
from datetime import datetime

from emergent.ship_abm.simulation_core import simulation


def point_to_polyline_distance(P, poly):
    # poly: list of (x,y) vertices
    px, py = P
    # defensive: if polygon has fewer than 2 vertices, return distance to that single point
    if poly is None or len(poly) == 0:
        return float('nan')
    if len(poly) == 1:
        ax, ay = poly[0]
        try:
            return float(math.hypot(px - ax, py - ay))
        except Exception:
            return float('nan')

    best = float('inf')
    for i in range(len(poly)-1):
        ax, ay = poly[i]
        bx, by = poly[i+1]
        # guard against NaN/Inf in waypoint coordinates
        if not np.isfinite(ax) or not np.isfinite(ay) or not np.isfinite(bx) or not np.isfinite(by):
            continue
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        denom = vx*vx + vy*vy
        if denom == 0:
            t = 0.0
        else:
            t = (vx*wx + vy*wy) / denom
            t = max(0.0, min(1.0, t))
        cx = ax + t * vx
        cy = ay + t * vy
        d = math.hypot(px - cx, py - cy)
        if d < best:
            best = d
    if not np.isfinite(best):
        return float('nan')
    return float(best)


def wrap_deg(d):
    return (d + 180) % 360 - 180


def heading_error_deg(psi_deg, cmd_deg):
    return wrap_deg(psi_deg - cmd_deg)


def run_one(wind_speed=0.5, dead_reck_sens=0.5, Kf_gain=0.002,
            Kp=0.5, Ki=0.05, Kd=0.12,
            max_rudder_deg=None,
            zig_legs=6, leg_length=200.0, zig_amp=30.0,
            dt=0.5, T=240.0, verbose=False, wind_fn_override=None,
            deriv_tau=None, backcalc_beta=None, max_rudder_rate=None):

    sim = simulation(port_name='Galveston', dt=dt, T=T, n_agents=1,
                     light_bg=True, verbose=verbose, use_ais=False, load_enc=False)

    # find a center start
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    # build zig-zag waypoints eastward, alternating north/south
    waypoints = []
    x0 = cx
    y0 = cy
    waypoints.append((x0, y0))
    direction = 1
    for i in range(1, zig_legs+1):
        x = x0 + i * leg_length
        y = y0 + direction * zig_amp
        waypoints.append((x, y))
        direction *= -1

    # supply waypoints in the shape expected by spawn(): a list per agent
    sim.waypoints = [waypoints]
    # spawn (will read sim.waypoints and initialize state accordingly)
    state0, pos0, psi0, goals = sim.spawn()

    # override tuning
    sim.ship.dead_reck_sensitivity = dead_reck_sens
    sim.ship.dead_reck_max_corr_deg = 30.0
    sim.ship.Kp = float(Kp)
    sim.ship.Ki = float(Ki)
    sim.ship.Kd = float(Kd)
    # optional overrides for derivative filter and anti-windup/backcalc
    try:
        from emergent.ship_abm.config import ADVANCED_CONTROLLER
        if deriv_tau is not None:
            sim.ship.deriv_tau = float(deriv_tau)
        if backcalc_beta is not None:
            ADVANCED_CONTROLLER['backcalc_beta'] = float(backcalc_beta)
        if max_rudder_rate is not None:
            sim.ship.max_rudder_rate = float(max_rudder_rate)
    except Exception:
        pass

    # override max rudder if requested
    if max_rudder_deg is not None:
        sim.ship.max_rudder = math.radians(float(max_rudder_deg))

    sim.tuning['Kf'] = float(Kf_gain)

    # deterministic northward wind (can be overridden by caller)
    def _default_wind_fn(lons, lats, when):
        import numpy as _np
        N = int(_np.atleast_1d(lons).size)
        return _np.tile(_np.array([[0.0, wind_speed]]), (N, 1))

    def _default_current_fn(lons, lats, when):
        import numpy as _np
        N = int(_np.atleast_1d(lons).size)
        return _np.tile(_np.array([[0.0, 0.0]]), (N, 1))

    sim.wind_fn = wind_fn_override if wind_fn_override is not None else _default_wind_fn
    sim.current_fn = _default_current_fn

    # run
    sim.run()

    # compute distance of each pos to polyline
    traj = np.array(sim.history[0])
    poly = waypoints
    dists = np.array([point_to_polyline_distance((px, py), poly) for (px, py) in traj])
    final_cross = float(dists[-1])
    rmse = float(np.sqrt(np.mean(dists**2)))

    # heading RMSE
    try:
        psi_arr = np.asarray(sim.psi_history)
        hd_cmd_arr = np.asarray(sim.hd_cmd_history)
        err_deg = heading_error_deg(np.degrees(psi_arr), np.degrees(hd_cmd_arr))
        heading_rmse = float(np.sqrt(np.mean(err_deg**2)))
    except Exception:
        heading_rmse = float('nan')

    # rudder stats
    try:
        rud_arr = np.asarray(getattr(sim, 'rudder_history', []))
        mean_abs_rudder = float(np.mean(np.abs(rud_arr))) if rud_arr.size else float('nan')
        rud_std = float(np.std(rud_arr)) if rud_arr.size else float('nan')
    except Exception:
        mean_abs_rudder = float('nan')
        rud_std = float('nan')

    # rudder saturation fraction: fraction of timesteps where |rudder| >= 0.98*max_rudder
    try:
        max_rudder = getattr(sim.ship, 'max_rudder', None)
        if rud_arr.size and max_rudder is not None and np.isfinite(max_rudder):
            sat_thresh = 0.98 * float(max_rudder)
            sat_frac = float(np.mean(np.abs(rud_arr) >= sat_thresh))
        else:
            sat_frac = float('nan')
    except Exception:
        sat_frac = float('nan')

    # settling time using heading error
    try:
        from emergent.ship_abm.simulation_core import settling_time, heading_error
        t_arr = np.asarray(sim.t_history)
        err_time_deg = heading_error(np.degrees(psi_arr), np.degrees(hd_cmd_arr))
        settle_s = float(settling_time(t_arr, err_time_deg, tol=2.0))
    except Exception:
        settle_s = float('nan')

    # composite cost
    w_pos = 1.0
    w_head = 0.5
    w_rudder = 0.1
    try:
        composite_cost = float(w_pos * rmse + w_head * heading_rmse + w_rudder * mean_abs_rudder)
    except Exception:
        composite_cost = float('nan')

    # write per-run CSV
    ts = os.getcwd()
    fname = f'zigzag_traj_w{wind_speed:.2f}_dr{dead_reck_sens:.2f}_Kf{Kf_gain:.4f}_Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}_mr{(max_rudder_deg if max_rudder_deg is not None else "cfg")}.csv'
    out_csv = os.path.join(ts, fname)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t','x_m','y_m','psi_deg','hd_cmd_deg','dist_to_poly_m','rudder_deg'])
        for i, t in enumerate(sim.t_history):
            pos = sim.history[0][i]
            psi = sim.psi_history[i]
            hd = sim.hd_cmd_history[i]
            rud = sim.rudder_history[i] if (hasattr(sim, 'rudder_history') and len(sim.rudder_history) > i) else float('nan')
            w.writerow([t, float(pos[0]), float(pos[1]), float(np.degrees(psi)), float(np.degrees(hd)), float(dists[i]), float(np.degrees(rud))])

    return {
        'wind_speed': wind_speed,
        'dead_reck_sens': dead_reck_sens,
        'Kf_gain': Kf_gain,
        'Kp': float(sim.ship.Kp),
        'Ki': float(sim.ship.Ki),
        'Kd': float(sim.ship.Kd),
        'final_cross_m': final_cross,
        'rmse_cross_m': rmse,
        'traj_csv': out_csv,
        'heading_rmse_deg': heading_rmse,
        'mean_abs_rudder_deg': float(np.degrees(mean_abs_rudder)) if not np.isnan(mean_abs_rudder) else float('nan'),
    'rudder_std_deg': float(np.degrees(rud_std)) if not np.isnan(rud_std) else float('nan'),
    'rudder_saturation_frac': float(sat_frac),
        'settling_s': settle_s,
        'composite_cost': composite_cost,
        'max_rudder_deg': float(max_rudder_deg) if max_rudder_deg is not None else float('nan')
    }


def main():
    # parameter grid defaults
    wind_speeds = [0.5, 1.5]
    dead_reck_sens_vals = [0.5]
    Kf_vals = [0.0, 0.002, 0.01, 0.02]
    # baseline tuned gains (from previous refinement)
    Kp = 0.5
    Ki = 0.05
    Kd = 0.12

    # increase max rudder for tests (deg)
    max_rudder_deg_override = 20.0

    results = []
    for wind_speed, dr_sens, Kf in [(w, d, k) for w in wind_speeds for d in dead_reck_sens_vals for k in Kf_vals]:
        print(f"[RUN] wind={wind_speed} m/s, dead_reck_sens={dr_sens}, Kf={Kf}, max_rudder_deg={max_rudder_deg_override}")
        res = run_one(wind_speed=wind_speed, dead_reck_sens=dr_sens, Kf_gain=Kf,
                      Kp=Kp, Ki=Ki, Kd=Kd,
                      max_rudder_deg=max_rudder_deg_override,
                      zig_legs=6, leg_length=200.0, zig_amp=30.0,
                      dt=0.5, T=240.0, verbose=False)
        print(f" -> rmse={res['rmse_cross_m']:.2f} m, final={res['final_cross_m']:.2f} m, traj={res['traj_csv']}")
        results.append(res)

    # write summary
    summary_path = os.path.join(os.getcwd(), 'zigzag_kf_sweep_summary.csv')
    with open(summary_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['wind_speed','dead_reck_sens','Kf_gain','Kp','Ki','Kd','final_cross_m','rmse_cross_m','heading_rmse_deg','mean_abs_rudder_deg','rudder_std_deg','rudder_saturation_frac','settling_s','composite_cost','max_rudder_deg','traj_csv'])
        for r in results:
            w.writerow([
                r['wind_speed'], r['dead_reck_sens'], r['Kf_gain'], r.get('Kp', float('nan')), r.get('Ki', float('nan')), r.get('Kd', float('nan')),
                r['final_cross_m'], r['rmse_cross_m'], r.get('heading_rmse_deg', float('nan')),
                r.get('mean_abs_rudder_deg', float('nan')), r.get('rudder_std_deg', float('nan')), r.get('rudder_saturation_frac', float('nan')),
                r.get('settling_s', float('nan')), r.get('composite_cost', float('nan')),
                r.get('max_rudder_deg', float('nan')), r['traj_csv']
            ])

    print('[SUMMARY] Wrote', summary_path)


if __name__ == '__main__':
    main()
