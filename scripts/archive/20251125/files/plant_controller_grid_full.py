"""Coordinated plant×controller full-length grid.

ndelta_mult ∈ [1.0,1.25,1.5]
Kp ∈ [0.35,0.40,0.45]
Kd = 0.50

Writes `sweep_results/plant_controller_grid_full.csv` and per-run traces.
"""
import os
import math
import time
import itertools
import pandas as pd
import numpy as np

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

NDELTA_MULTS = [1.0, 1.25, 1.5]
KP_LIST = [0.35, 0.40, 0.45]
KD = 0.50
KI = 0.0

# full run settings
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
    for mult, Kp in itertools.product(NDELTA_MULTS, KP_LIST):
        tag = f'nd{mult:.2f}_Kp{Kp:.3f}_Kd{KD:.3f}_fullgrid'
        trace_path = os.path.join(OUT, f'pc_grid_{tag}.csv')
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = trace_path

        config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * mult
        config.CONTROLLER_GAINS['Kp'] = float(Kp)
        config.CONTROLLER_GAINS['Ki'] = float(KI)
        config.CONTROLLER_GAINS['Kd'] = float(KD)

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

        rows.append({'ndelta_mult': mult, 'Kp': Kp, 'Kd': KD, 'trace': trace_path,
                     'max_err_deg': max_err, 'mean_err_deg': mean_err,
                     'max_raw_deg': max_raw, 'max_rud_deg': max_rud,
                     'sat_frac': sat_frac, 'run_time_s': dur})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'plant_controller_grid_full.csv'), index=False)
        print('Completed', tag, 'mean_err_deg=', mean_err)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')