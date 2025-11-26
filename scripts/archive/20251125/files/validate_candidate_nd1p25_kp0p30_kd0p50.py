"""Run straight-line crosswind and zigzag crosswind validations for a candidate.

Writes two summary CSVs under sweep_results/:
- straight_candidate_summary.csv
- zigzag_candidate_summary.csv
"""
import os
import math
import time
import numpy as np
import pandas as pd

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

DT = 0.1
T = 300.0
WIND_SPEED = 5.0

ndelta_mult = 1.25
Kp = 0.30
Ki = 0.0
Kd = 0.50
rudder_tau = 1.0

def constant_crosswind_for_sim(sim, wind_speed):
    psi0 = float(sim.psi[0])
    cross_theta = psi0 + math.pi / 2.0
    wx = wind_speed * math.cos(cross_theta)
    wy = wind_speed * math.sin(cross_theta)

    def wind_fn(lon, lat, when):
        return np.tile(np.array([[wx, wy]]), (1, 1))

    return wind_fn

orig_ship = config.SHIP_PHYSICS.copy()
orig_ctrl = config.CONTROLLER_GAINS.copy()
orig_adv = config.ADVANCED_CONTROLLER.copy()

try:
    config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * ndelta_mult
    config.CONTROLLER_GAINS['Kp'] = float(Kp)
    config.CONTROLLER_GAINS['Ki'] = float(Ki)
    config.CONTROLLER_GAINS['Kd'] = float(Kd)

    # Straight-line crosswind
    trace1 = os.path.join(OUT, 'pid_trace_validate_straight_nd1p25.csv')
    config.PID_TRACE['enabled'] = True
    config.PID_TRACE['path'] = trace1
    sim1 = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                      test_mode='straight',)
    # minimal route for spawn: provide waypoints directly
    try:
        sim1.waypoints = [[(-94.8, 29.3), (-94.7, 29.35)]]
    except Exception:
        pass
    sim1.spawn()
    sim1.wind_fn = constant_crosswind_for_sim(sim1, WIND_SPEED)
    sim1.current_fn = lambda lon, lat, when: np.zeros((1, 2))
    try:
        sim1.ship.rudder_tau = float(rudder_tau)
    except Exception:
        pass
    start = time.time()
    sim1.run()
    dur1 = time.time() - start

    df1 = pd.read_csv(trace1)
    df10 = df1[df1['agent'] == 0]
    straight_summary = {
        'trace': trace1,
        'max_err_deg': float(df10['err_deg'].abs().max()),
        'mean_err_deg': float(df10['err_deg'].abs().mean()),
        'max_rud_deg': float(df10['rud_deg'].abs().max()),
        'sat_frac': float((df10['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df10)),
        'run_time_s': dur1
    }
    pd.DataFrame([straight_summary]).to_csv(os.path.join(OUT, 'straight_candidate_summary.csv'), index=False)

    # Zigzag crosswind
    trace2 = os.path.join(OUT, 'pid_trace_validate_zigzag_nd1p25.csv')
    config.PID_TRACE['enabled'] = True
    config.PID_TRACE['path'] = trace2
    sim2 = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                      test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
    try:
        sim2.waypoints = [[(-94.8, 29.3), (-94.7, 29.35)]]
    except Exception:
        pass
    sim2.spawn()
    sim2.wind_fn = constant_crosswind_for_sim(sim2, WIND_SPEED)
    sim2.current_fn = lambda lon, lat, when: np.zeros((1, 2))
    try:
        sim2.ship.rudder_tau = float(rudder_tau)
    except Exception:
        pass
    start = time.time()
    sim2.run()
    dur2 = time.time() - start

    df2 = pd.read_csv(trace2)
    df20 = df2[df2['agent'] == 0]
    zigzag_summary = {
        'trace': trace2,
        'max_err_deg': float(df20['err_deg'].abs().max()),
        'mean_err_deg': float(df20['err_deg'].abs().mean()),
        'max_rud_deg': float(df20['rud_deg'].abs().max()),
        'sat_frac': float((df20['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df20)),
        'run_time_s': dur2
    }
    pd.DataFrame([zigzag_summary]).to_csv(os.path.join(OUT, 'zigzag_candidate_summary.csv'), index=False)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')
