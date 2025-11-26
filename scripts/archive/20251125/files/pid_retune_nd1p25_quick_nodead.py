"""Quick PID retune grid at Ndelta multiplier = 1.25 with dead-reck disabled.

Runs a 3x3 grid (Kp x Kd) and writes `sweep_results/pid_retune_nd1p25_summary_quick_nodead.csv`.
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

# grid
KP_LIST = [0.30, 0.35, 0.40]
KD_LIST = [0.30, 0.40, 0.50]
KI = 0.0

# sim settings (shortened for quick runs)
DT = 0.1
T = 60.0
ZIGDEG = 15
ZIGHOLD = 30
WIND_SPEED = 3.0

ndelta_mult = 1.25
rudder_tau = 1.0

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

    for Kp, Kd in itertools.product(KP_LIST, KD_LIST):
        tag = f'nd{ndelta_mult:.2f}_Kp{Kp:.3f}_Kd{Kd:.3f}_nodead'
        trace_path = os.path.join(OUT, f'pid_trace_{tag}_quick.csv')
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = trace_path

        config.CONTROLLER_GAINS['Kp'] = float(Kp)
        config.CONTROLLER_GAINS['Ki'] = float(KI)
        config.CONTROLLER_GAINS['Kd'] = float(Kd)

        print('Running', tag)

        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=ZIGDEG, zigzag_hold=ZIGHOLD)
        sim.spawn()

        # disable dead-reck per-ship flag
        try:
            sim.ship.disable_dead_reck = True
        except Exception:
            pass

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

        rows.append({'ndelta_mult': ndelta_mult, 'rudder_tau': rudder_tau,
                     'Kp': Kp, 'Ki': KI, 'Kd': Kd, 'trace': trace_path,
                     'max_err_deg': max_err, 'mean_err_deg': mean_err,
                     'max_raw_deg': max_raw, 'max_rud_deg': max_rud,
                     'sat_frac': sat_frac, 'run_time_s': dur})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pid_retune_nd1p25_summary_quick_nodead.csv'), index=False)
        print('Completed', tag, 'mean_err_deg=', mean_err)

finally:
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)
    print('Restored original config')
