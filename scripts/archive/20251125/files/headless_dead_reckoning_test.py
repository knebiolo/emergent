"""
Headless dead-reckoning test

Spawns a single vessel and sends it toward a waypoint 1 km east while
exposing a constant crosswind from the south (northward wind). Prints
summary statistics and a small CSV of the trajectory.
"""
import numpy as np
import itertools
import csv, os
from datetime import datetime

# instantiate simulation class
from emergent.ship_abm.simulation_core import simulation


def run_experiment(wind_speed=3.0, dead_reck_sens=0.25, Kf_gain=0.002, dt=0.5, T=300.0, verbose=False, Kp=None, Ki=None, Kd=None):
    sim = simulation(
        port_name='Galveston',
        dt=dt,
        T=T,
        n_agents=1,
        light_bg=True,
        verbose=verbose,
        use_ais=False,
        load_enc=False,
        test_mode=None
    )

    # start/goal
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    start = (cx, cy)
    goal = (cx + 1000.0, cy)
    sim.waypoints = [[start, goal]]
    state0, pos0, psi0, goals = sim.spawn()

    # override ship tuning values
    sim.ship.dead_reck_sensitivity = dead_reck_sens
    sim.ship.dead_reck_max_corr_deg = 30.0
    # optional PID overrides
    if Kp is not None:
        sim.ship.Kp = float(Kp)
    if Ki is not None:
        sim.ship.Ki = float(Ki)
    if Kd is not None:
        sim.ship.Kd = float(Kd)
    # update advanced controller feed-forward via sim.tuning
    sim.tuning['Kf'] = Kf_gain
    sim.tuning['r_rate_max_deg'] = 3.0

    # deterministic samplers
    def wind_fn(lons, lats, when):
        import numpy as _np
        N = int(_np.atleast_1d(lons).size)
        # return array shape (N,2) -> [east_comp, north_comp]
        return _np.tile(_np.array([[0.0, wind_speed]]), (N, 1))

    def current_fn(lons, lats, when):
        import numpy as _np
        N = int(_np.atleast_1d(lons).size)
        return _np.tile(_np.array([[0.0, 0.0]]), (N, 1))

    sim.wind_fn = wind_fn
    sim.current_fn = current_fn

    # run
    sim.run()

    # compute cross-track error time series relative to the line from start->goal
    traj = np.array(sim.history[0])  # Nx2
    # line vector eastwards
    A = np.array(start)
    B = np.array(goal)
    AB = B - A
    AB_unit = AB / (np.linalg.norm(AB) + 1e-12)
    # cross-track error = projection of (P - A) onto normal
    normal = np.array([-AB_unit[1], AB_unit[0]])
    ct_errors = np.dot(traj - A, normal)
    final_cross = ct_errors[-1]
    rmse = np.sqrt(np.mean(ct_errors**2))

    # --- additional metrics: heading RMSE, mean abs rudder, rudder std, settling time
    try:
        psi_arr = np.asarray(sim.psi_history)  # radians
        hd_cmd_arr = np.asarray(sim.hd_cmd_history)  # radians
        # heading error (deg) canonical wrapped
        from emergent.ship_abm.angle_utils import heading_diff_deg
        err_deg = heading_diff_deg(np.degrees(hd_cmd_arr), np.degrees(psi_arr))
        heading_rmse = float(np.sqrt(np.mean(err_deg**2)))
    except Exception:
        heading_rmse = float('nan')

    try:
        rud_arr = np.asarray(getattr(sim, 'rudder_history', []))  # radians
        mean_abs_rudder = float(np.mean(np.abs(rud_arr))) if rud_arr.size else float('nan')
        rudder_std = float(np.std(rud_arr)) if rud_arr.size else float('nan')
    except Exception:
        mean_abs_rudder = float('nan')
        rudder_std = float('nan')

    # settling time for heading within ±2° of commanded
    try:
        from emergent.ship_abm.simulation_core import settling_time, heading_error
        t_arr = np.asarray(sim.t_history)
        # heading_error expects actual_deg, commanded_deg
        err_time_deg = heading_error(np.degrees(psi_arr), np.degrees(hd_cmd_arr))
        settle_s = float(settling_time(t_arr, err_time_deg, tol=2.0))
    except Exception:
        settle_s = float('nan')

    # composite cost: weight position RMSE, heading RMSE, and mean rudder effort
    # weights are configurable; these defaults prefer tracking accuracy
    w_pos = 1.0
    w_head = 0.5
    w_rudder = 0.1
    try:
        composite_cost = float(w_pos * rmse + w_head * heading_rmse + w_rudder * mean_abs_rudder)
    except Exception:
        composite_cost = float('nan')

    # save trajectory CSV per run
    ts = os.getcwd()
    fname = f'headless_dead_reckoning_traj_w{wind_speed:.2f}_dr{dead_reck_sens:.2f}_Kf{Kf_gain:.4f}.csv'
    out_csv = os.path.join(ts, fname)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t','x_m','y_m','psi_deg','hd_cmd_deg','cross_track_m','rudder_deg'])
        for i, t in enumerate(sim.t_history):
            pos = sim.history[0][i]
            psi = sim.psi_history[i]
            hd = sim.hd_cmd_history[i]
            rud = sim.rudder_history[i] if (hasattr(sim, 'rudder_history') and len(sim.rudder_history) > i) else float('nan')
            w.writerow([t, float(pos[0]), float(pos[1]), float(np.degrees(psi)), float(np.degrees(hd)), float(ct_errors[i]), float(np.degrees(rud))])

    return {
        'wind_speed': wind_speed,
        'dead_reck_sens': dead_reck_sens,
        'Kf_gain': Kf_gain,
        'Kp': float(sim.ship.Kp),
        'Ki': float(sim.ship.Ki),
        'Kd': float(sim.ship.Kd),
        'final_cross_m': float(final_cross),
        'rmse_cross_m': float(rmse),
        'traj_csv': out_csv,
        'heading_rmse_deg': heading_rmse,
        'mean_abs_rudder_deg': float(np.degrees(mean_abs_rudder)) if not np.isnan(mean_abs_rudder) else float('nan'),
        'rudder_std_deg': float(np.degrees(rudder_std)) if not np.isnan(rudder_std) else float('nan'),
        'settling_s': settle_s,
        'composite_cost': composite_cost
    }


def main():
    # parameter grid
    wind_speeds = [0.5, 1.5, 3.0]
    dead_reck_sens_vals = [0.1, 0.25, 0.5]
    Kf_vals = [0.0, 0.002, 0.01]

    results = []
    for wind_speed, dr_sens, Kf in itertools.product(wind_speeds, dead_reck_sens_vals, Kf_vals):
        print(f"[RUN] wind={wind_speed} m/s, dead_reck_sens={dr_sens}, Kf={Kf}")
        res = run_experiment(wind_speed=wind_speed, dead_reck_sens=dr_sens, Kf_gain=Kf, dt=0.5, T=300.0, verbose=False)
        print(f" -> final_cross={res['final_cross_m']:.2f} m, rmse={res['rmse_cross_m']:.2f} m, traj={res['traj_csv']}")
        results.append(res)

    # write summary
    summary_path = os.path.join(os.getcwd(), 'dead_reckoning_experiments_summary.csv')
    with open(summary_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['wind_speed','dead_reck_sens','Kf_gain','Kp','Ki','Kd','final_cross_m','rmse_cross_m','heading_rmse_deg','mean_abs_rudder_deg','rudder_std_deg','settling_s','composite_cost','traj_csv'])
        for r in results:
            w.writerow([
                r['wind_speed'], r['dead_reck_sens'], r['Kf_gain'], r.get('Kp', float('nan')), r.get('Ki', float('nan')), r.get('Kd', float('nan')),
                r['final_cross_m'], r['rmse_cross_m'], r.get('heading_rmse_deg', float('nan')),
                r.get('mean_abs_rudder_deg', float('nan')), r.get('rudder_std_deg', float('nan')),
                r.get('settling_s', float('nan')), r.get('composite_cost', float('nan')), r['traj_csv']
            ])

    print('[SUMMARY] Wrote', summary_path)


if __name__ == '__main__':
    main()
