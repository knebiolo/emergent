"""Resume PID retune grid for ndelta_mult=1.25: run only missing combos.

Reads existing sweep_results/pid_retune_nd1p25_summary.csv (if any) and runs the remaining
(Kp, Kd) combos, appending results.
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

KP_LIST = [0.30, 0.35, 0.40]
KD_LIST = [0.30, 0.40, 0.50]
KI = 0.0

DT = 0.1
T = 300.0
ZIGDEG = 15
ZIGHOLD = 30
WIND_SPEED = 5.0

ndelta_mult = 1.25
rudder_tau = 1.0

summary_path = os.path.join(OUT, 'pid_retune_nd1p25_summary.csv')
completed = set()
rows = []
if os.path.exists(summary_path):
    try:
        df = pd.read_csv(summary_path)
        for _, r in df.iterrows():
            kp = float(r['Kp'])
            kd = float(r['Kd'])
            completed.add((kp, kd))
            rows.append(r.to_dict())
    except Exception:
        pass

orig_ship = config.SHIP_PHYSICS.copy()
orig_ctrl = config.CONTROLLER_GAINS.copy()
orig_adv = config.ADVANCED_CONTROLLER.copy()

try:
    config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * ndelta_mult

    for Kp, Kd in itertools.product(KP_LIST, KD_LIST):
        if (Kp, Kd) in completed:
            print('Skipping completed Kp, Kd', Kp, Kd)
            continue

        tag = f'nd{ndelta_mult:.2f}_Kp{Kp:.3f}_Kd{Kd:.3f}'
        trace_path = os.path.join(OUT, f'pid_trace_{tag}.csv')
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = trace_path

        config.CONTROLLER_GAINS['Kp'] = float(Kp)
        config.CONTROLLER_GAINS['Ki'] = float(KI)
        config.CONTROLLER_GAINS['Kd'] = float(Kd)

        print('Running', tag)

        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=ZIGDEG, zigzag_hold=ZIGHOLD)
        sim.spawn()

        try:
            sim.ship.rudder_tau = float(rudder_tau)
        except Exception:
            pass
        sim.tuning['Kp'] = float(sim.tuning.get('Kp', sim.ship.Kp))
        sim.tuning['Ki'] = float(sim.tuning.get('Ki', sim.ship.Ki))
        sim.tuning['Kd'] = float(sim.tuning.get('Kd', sim.ship.Kd))
        sim.tuning['deriv_tau'] = float(sim.tuning.get('deriv_tau', sim.ship.deriv_tau))
        sim.tuning['lead_time'] = float(sim.tuning.get('lead_time', sim.ship.lead_time))
        sim.tuning['release_band_deg'] = float(sim.tuning.get('release_band_deg', math.degrees(sim.ship.release_band_deg)))

        sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[WIND_SPEED * math.cos(float(sim.psi[0]) + math.pi/2.0), WIND_SPEED * math.sin(float(sim.psi[0]) + math.pi/2.0)]]),(1,1))
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

        df_run = pd.read_csv(trace_path)
        df0 = df_run[df_run['agent'] == 0]

        max_err = float(df0['err_deg'].abs().max())
        mean_err = float(df0['err_deg'].abs().mean())
        max_raw = float(df0['raw_deg'].abs().max())
        max_rud = float(df0['rud_deg'].abs().max())
        sat_frac = float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

        row = {'ndelta_mult': ndelta_mult, 'rudder_tau': rudder_tau,
               'Kp': Kp, 'Ki': KI, 'Kd': Kd, 'trace': trace_path,
               'max_err_deg': max_err, 'mean_err_deg': mean_err,
               'max_raw_deg': max_raw, 'max_rud_deg': max_rud,
               'sat_frac': sat_frac, 'run_time_s': dur}

        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print('Completed', tag, 'mean_err_deg=', mean_err)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')
