"""Headless zigzag under steady perpendicular crosswind.

Runs simulation in test_mode='zigzag', attaches a constant crosswind (starboard),
writes PID trace CSV, computes zigzag metrics, saves top-down track and time-series
plots, and writes a summary CSV.

Usage: python scripts/zigzag_crosswind_headless.py
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics
from emergent.ship_abm.config import PID_TRACE, SHIP_PHYSICS

DT = 0.1
T = 400.0
WIND_SPEED = 5.0
TRACE_PATH = 'scripts/pid_trace_zigzag_crosswind.csv'
OUT_TRACK = 'scripts/zigzag_crosswind_track.csv'
OUT_SUM = 'scripts/zigzag_crosswind_summary.csv'
PLOT_DIR = 'plots'

def make_constant_crosswind(sim, wind_speed):
    # compute a perpendicular (starboard) wind relative to initial heading after spawn
    psi0 = float(sim.psi[0])
    cross_theta = psi0 + math.pi/2.0
    wx = wind_speed * math.cos(cross_theta)
    wy = wind_speed * math.sin(cross_theta)

    def wind_fn(lon, lat, when):
        return np.tile(np.array([[wx, wy]]), (1,1))

    return wind_fn


def run():
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = TRACE_PATH

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)

    # prepare waypoints/spawn then set wind (make_constant_crosswind uses sim.psi)
    sim.spawn()
    sim.wind_fn = make_constant_crosswind(sim, WIND_SPEED)
    sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

    # run
    print(f"Running zigzag crosswind T={T}s dt={DT}s, wind={WIND_SPEED} m/s")
    if os.path.exists(TRACE_PATH):
        try:
            os.remove(TRACE_PATH)
        except Exception:
            pass

    sim.run()

    # collect histories
    hist = getattr(sim, 'history', {})
    seq = hist.get(0, [])
    if len(seq) == 0:
        raise RuntimeError('No trajectory recorded')
    xs = [p[0] for p in seq]
    ys = [p[1] for p in seq]
    n = min(len(xs), len(ys), len(sim.t_history))
    if n == 0:
        raise RuntimeError('No synchronized trajectory/time data available')
    xs = xs[:n]
    ys = ys[:n]
    t = np.array(sim.t_history[:n])

    os.makedirs('scripts', exist_ok=True)
    pd.DataFrame({'t': t, 'x': xs, 'y': ys}).to_csv(OUT_TRACK, index=False)
    print('Wrote', OUT_TRACK)

    # read pid trace
    df = pd.read_csv(TRACE_PATH)
    df0 = df[df['agent'] == 0]

    os.makedirs(PLOT_DIR, exist_ok=True)
    # top-down track
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, '-k')
    plt.scatter(xs[0], ys[0], c='g')
    plt.scatter(xs[-1], ys[-1], c='r')
    plt.axis('equal')
    plt.title(f'Zigzag crosswind track (w={WIND_SPEED} m/s)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    track_png = os.path.join(PLOT_DIR, 'zigzag_crosswind_track.png')
    plt.savefig(track_png)
    plt.close()
    print('Wrote', track_png)

    # heading & error
    plt.figure(figsize=(10,4))
    plt.plot(df0['t'], df0['psi_deg'], label='psi (deg)')
    plt.plot(df0['t'], df0['err_deg'], label='err_deg')
    plt.grid(True)
    plt.legend()
    head_png = os.path.join(PLOT_DIR, 'zigzag_crosswind_heading.png')
    plt.savefig(head_png)
    plt.close()
    print('Wrote', head_png)

    # rudder
    plt.figure(figsize=(10,4))
    plt.plot(df0['t'], df0['raw_deg'], '--', label='raw cmd (deg)', alpha=0.7)
    plt.plot(df0['t'], df0['rud_deg'], label='applied rudder (deg)')
    plt.grid(True)
    plt.legend()
    rud_png = os.path.join(PLOT_DIR, 'zigzag_crosswind_rudder.png')
    plt.savefig(rud_png)
    plt.close()
    print('Wrote', rud_png)

    # metrics (use compute_zigzag_metrics)
    try:
        actual_heading = np.array(sim.psi_history)
        cmd_heading = np.array(sim.hd_cmd_history)
        metrics = compute_zigzag_metrics(np.array(sim.t_history[:len(actual_heading)]), actual_heading, cmd_heading, tol=5.0)
    except Exception as e:
        metrics = {'error': str(e)}

    # numeric summary
    max_err = float(df0['err_deg'].abs().max())
    mean_err = float(df0['err_deg'].abs().mean())
    max_raw = float(df0['raw_deg'].abs().max())
    max_rud = float(df0['rud_deg'].abs().max())
    sat_frac = float((df0['rud_deg'].abs() >= math.degrees(SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

    summary = {
        'wind_speed_m_s': WIND_SPEED,
        'max_err_deg': max_err,
        'mean_err_deg': mean_err,
        'max_raw_deg': max_raw,
        'max_rud_deg': max_rud,
        'sat_frac': sat_frac,
        'zigzag_metrics': str(metrics)
    }
    pd.DataFrame([summary]).to_csv(OUT_SUM, index=False)
    print('Wrote', OUT_SUM)
    print('Summary:', summary)


if __name__ == '__main__':
    run()
