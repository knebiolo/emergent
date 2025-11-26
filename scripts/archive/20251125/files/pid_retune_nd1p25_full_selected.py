"""Run full-length PID validation for selected candidate combos.

Writes `sweep_results/pid_retune_nd1p25_summary_selected.csv`.
"""
import os
import math
import time
import pandas as pd
import numpy as np

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

CASES = [
    (0.40, 0.50),
    (0.40, 0.40),
    (0.35, 0.50),
]

ndelta_mult = 1.25
rudder_tau = 1.0

# long run settings
DT = 0.1
T = 300.0
ZIGDEG = 15
ZIGHOLD = 30
WIND_SPEED = 3.0

rows = []

orig_ship = config.SHIP_PHYSICS.copy()
orig_ctrl = config.CONTROLLER_GAINS.copy()
orig_adv = config.ADVANCED_CONTROLLER.copy()

try:
    config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * ndelta_mult

    for Kp, Kd in CASES:
        tag = f'nd{ndelta_mult:.2f}_Kp{Kp:.3f}_Kd{Kd:.3f}_full'
        trace_path = os.path.join(OUT, f'pid_trace_{tag}.csv')
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = trace_path

        config.CONTROLLER_GAINS['Kp'] = float(Kp)
        config.CONTROLLER_GAINS['Ki'] = float(0.0)
        config.CONTROLLER_GAINS['Kd'] = float(Kd)

        print('Running', tag)

        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=ZIGDEG, zigzag_hold=ZIGHOLD)
        sim.spawn()

        sim.wind_fn = lambda lon, lat, when: (3.0 * np.array([[0.0, 1.0]]))
        sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))

        start = time.time()
        sim.run()
        dur = time.time() - start

        if not os.path.exists(trace_path):
            print('Warning: missing trace for', tag)
            continue

        df = pd.read_csv(trace_path)
        df0 = df[df['agent'] == 0]

        max_err = float(df0['err_deg'].abs().max())
        mean_err = float(df0['err_deg'].abs().mean())
        max_raw = float(df0['raw_deg'].abs().max())
        max_rud = float(df0['rud_deg'].abs().max())
        sat_frac = float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

        rows.append({'ndelta_mult': ndelta_mult, 'rudder_tau': rudder_tau,
                     'Kp': Kp, 'Ki': 0.0, 'Kd': Kd, 'trace': trace_path,
                     'max_err_deg': max_err, 'mean_err_deg': mean_err,
                     'max_raw_deg': max_raw, 'max_rud_deg': max_rud,
                     'sat_frac': sat_frac, 'run_time_s': dur})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pid_retune_nd1p25_summary_selected.csv'), index=False)
        print('Completed', tag, 'mean_err_deg=', mean_err)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')