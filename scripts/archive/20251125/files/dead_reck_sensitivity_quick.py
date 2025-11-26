"""Quick sweep over dead-reck parameters (lead_time x max_corr_deg).

Writes `sweep_results/dead_reck_sensitivity_quick.csv` and per-run traces.
"""
import os
import math
import time
import itertools
import numpy as np
import pandas as pd

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

# sweep ranges (user confirmed)
LEAD_TIMES = [1.0, 3.0, 5.0]
MAX_CORR_DEGS = [5, 10, 15]

# PID and plant settings
Kp = 0.40
Ki = 0.0
Kd = 0.50
ndelta_mult = 1.25

# sim settings (quick)
DT = 0.1
T = 60.0
ZIGDEG = 15
ZIGHOLD = 30
WIND_SPEED = 3.0

rows = []

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

    for lead_time, max_corr_deg in itertools.product(LEAD_TIMES, MAX_CORR_DEGS):
        tag = f'lt{lead_time:.1f}_mc{max_corr_deg:d}'
        trace_path = os.path.join(OUT, f'dr_sens_{tag}.csv')
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = trace_path

        # update gains
        config.CONTROLLER_GAINS['Kp'] = float(Kp)
        config.CONTROLLER_GAINS['Ki'] = float(Ki)
        config.CONTROLLER_GAINS['Kd'] = float(Kd)

        # set dead-reck parameters via ADVANCED_CONTROLLER
        config.ADVANCED_CONTROLLER['lead_time'] = float(lead_time)
        config.ADVANCED_CONTROLLER['dead_reck_max_corr_deg'] = float(max_corr_deg)

        print('Running', tag)

        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=ZIGDEG, zigzag_hold=ZIGHOLD)
        sim.spawn()

        # ensure ship-side tuning reflects config
        sim.tuning['Kp'] = float(sim.tuning.get('Kp', sim.ship.Kp))
        sim.tuning['Ki'] = float(sim.tuning.get('Ki', sim.ship.Ki))
        sim.tuning['Kd'] = float(sim.tuning.get('Kd', sim.ship.Kd))
        sim.tuning['lead_time'] = float(sim.tuning.get('lead_time', sim.ship.lead_time))

        sim.wind_fn = constant_crosswind_for_sim(sim, WIND_SPEED)
        sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))

        try:
            if os.path.exists(trace_path):
                os.remove(trace_path)
        except Exception:
            pass

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

        rows.append({'lead_time': lead_time, 'max_corr_deg': max_corr_deg,
                     'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'trace': trace_path,
                     'max_err_deg': max_err, 'mean_err_deg': mean_err,
                     'max_raw_deg': max_raw, 'max_rud_deg': max_rud,
                     'sat_frac': sat_frac, 'run_time_s': dur})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'dead_reck_sensitivity_quick.csv'), index=False)
        print('Completed', tag, 'mean_err_deg=', mean_err)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')