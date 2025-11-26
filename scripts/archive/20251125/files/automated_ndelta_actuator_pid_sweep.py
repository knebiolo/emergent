"""Automated two-phase sweep:

Phase 1: sweep Ndelta multipliers × actuator params × feed-forward gain (modest grid)
Phase 2: pick top candidates and run a small PID refinement grid around each

Writes per-run PID traces and two summary CSVs under `sweep_results/`.

Usage: python scripts/automated_ndelta_actuator_pid_sweep.py
"""
import os
import math
import time
import itertools
import numpy as np
import pandas as pd

from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics
from emergent.ship_abm import config


OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters for the sweep (kept deliberately small)
NDELTA_MULTS = [1.0, 1.25, 1.5]
MAX_RUDDER_RATE_LIST = [config.SHIP_PHYSICS['max_rudder_rate'], 0.17]  # keep current and faster
RUDDER_TAU_LIST = [1.0, 0.5]
KF_LIST = [config.ADVANCED_CONTROLLER.get('Kf_gain', 0.0), 0.02]

# PID refinement grid (phase 2)
KP_LIST = [0.5, 0.8, 1.0]
KI_LIST = [0.0, 0.02]
KD_LIST = [0.12, 0.3]

# Simulation settings
DT = 0.1
T_PHASE1 = 300.0
T_PHASE2 = 300.0
WIND_SPEED = 5.0

# bookkeeping
phase1_rows = []
phase2_rows = []

# save originals to restore
orig_ship = config.SHIP_PHYSICS.copy()
orig_ctrl = config.CONTROLLER_GAINS.copy()
orig_adv = config.ADVANCED_CONTROLLER.copy()


def constant_crosswind_for_sim(sim, wind_speed):
    psi0 = float(sim.psi[0])
    cross_theta = psi0 + math.pi / 2.0
    wx = wind_speed * math.cos(cross_theta)
    wy = wind_speed * math.sin(cross_theta)

    def wind_fn(lon, lat, when):
        return np.tile(np.array([[wx, wy]]), (1, 1))

    return wind_fn


def run_one(cfg_tag, T, zigzag_deg, zigzag_hold):
    trace_path = os.path.join(OUT_DIR, f'pid_trace_{cfg_tag}.csv')
    config.PID_TRACE['enabled'] = True
    config.PID_TRACE['path'] = trace_path

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=zigzag_deg, zigzag_hold=zigzag_hold)
    # spawn then set wind
    sim.spawn()
    sim.wind_fn = constant_crosswind_for_sim(sim, WIND_SPEED)
    sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))

    # remove any old trace
    try:
        if os.path.exists(trace_path):
            os.remove(trace_path)
    except Exception:
        pass

    start = time.time()
    sim.run()
    dur = time.time() - start

    # read pid trace
    if not os.path.exists(trace_path):
        raise RuntimeError('Expected pid trace not found: ' + trace_path)

    df = pd.read_csv(trace_path)
    df0 = df[df['agent'] == 0]

    # basic metrics
    max_err = float(df0['err_deg'].abs().max())
    mean_err = float(df0['err_deg'].abs().mean())
    max_raw = float(df0['raw_deg'].abs().max())
    max_rud = float(df0['rud_deg'].abs().max())
    sat_frac = float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

    # zigzag metrics if available from sim histories
    try:
        t_arr = np.array(sim.t_history)
        psi_arr = np.array(sim.psi_history)
        hd_arr = np.array(sim.hd_cmd_history)
        zz = compute_zigzag_metrics(t_arr, psi_arr, hd_arr, tol=5.0)
    except Exception:
        zz = {}

    return {
        'trace': trace_path,
        'max_err_deg': max_err,
        'mean_err_deg': mean_err,
        'max_raw_deg': max_raw,
        'max_rud_deg': max_rud,
        'sat_frac': sat_frac,
        'zigzag_metrics': str(zz),
        'run_time_s': dur
    }


def phase1():
    print('Starting phase 1: Ndelta × actuator × Kf sweep')
    combos = list(itertools.product(NDELTA_MULTS, MAX_RUDDER_RATE_LIST, RUDDER_TAU_LIST, KF_LIST))
    for mult, mr, tau, kf in combos:
        # set config
        config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * mult
        config.SHIP_PHYSICS['max_rudder_rate'] = float(mr)
        # max_rudder stays the same per user request
        # ship model's rudder_tau is set on ship instance; we will set it after sim.spawn
        config.ADVANCED_CONTROLLER['Kf_gain'] = float(kf)

        tag = f'nd{mult:.2f}_mr{mr:.3f}_tau{tau:.2f}_kf{kf:.4f}'
        print('Running', tag)

        # run sim
        # create sim then set ship.rudder_tau before run
        config.PID_TRACE['enabled'] = True
        config.PID_TRACE['path'] = os.path.join(OUT_DIR, f'pid_trace_{tag}.csv')
        sim = simulation(port_name='Galveston', dt=DT, T=T_PHASE1, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
        sim.spawn()
        # set actuator tau on the ship object
        try:
            sim.ship.rudder_tau = float(tau)
        except Exception:
            pass
        sim.wind_fn = constant_crosswind_for_sim(sim, WIND_SPEED)
        sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))

        # ensure old trace removed
        try:
            if os.path.exists(config.PID_TRACE['path']):
                os.remove(config.PID_TRACE['path'])
        except Exception:
            pass

        start = time.time()
        sim.run()
        dur = time.time() - start

        if not os.path.exists(config.PID_TRACE['path']):
            print('Warning: missing trace for', tag)
            continue

        df = pd.read_csv(config.PID_TRACE['path'])
        df0 = df[df['agent'] == 0]
        max_err = float(df0['err_deg'].abs().max())
        mean_err = float(df0['err_deg'].abs().mean())
        max_raw = float(df0['raw_deg'].abs().max())
        max_rud = float(df0['rud_deg'].abs().max())
        sat_frac = float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

        row = {
            'ndelta_mult': mult,
            'max_rudder_rate': mr,
            'rudder_tau': tau,
            'Kf_gain': kf,
            'trace': config.PID_TRACE['path'],
            'max_err_deg': max_err,
            'mean_err_deg': mean_err,
            'max_raw_deg': max_raw,
            'max_rud_deg': max_rud,
            'sat_frac': sat_frac,
            'run_time_s': dur
        }
        phase1_rows.append(row)
        # save incremental summary
        pd.DataFrame(phase1_rows).to_csv(os.path.join(OUT_DIR, 'ndelta_actuator_summary.csv'), index=False)

    print('Phase 1 complete')


def phase2(top_k=3):
    print('Starting phase 2: PID refinement around top candidates')
    # choose top candidates by mean_err
    df1 = pd.DataFrame(phase1_rows)
    df1 = df1.sort_values('mean_err_deg')
    top = df1.head(top_k)

    for _, r in top.iterrows():
        mult = float(r['ndelta_mult'])
        mr = float(r['max_rudder_rate'])
        tau = float(r['rudder_tau'])
        kf = float(r['Kf_gain'])

        # set plant/actuator as in candidate
        config.SHIP_PHYSICS['Ndelta'] = orig_ship['Ndelta'] * mult
        config.SHIP_PHYSICS['max_rudder_rate'] = mr
        config.ADVANCED_CONTROLLER['Kf_gain'] = kf

        for Kp, Ki, Kd in itertools.product(KP_LIST, KI_LIST, KD_LIST):
            # set PID
            config.CONTROLLER_GAINS['Kp'] = float(Kp)
            config.CONTROLLER_GAINS['Ki'] = float(Ki)
            config.CONTROLLER_GAINS['Kd'] = float(Kd)

            tag = f'nd{mult:.2f}_mr{mr:.3f}_tau{tau:.2f}_kf{kf:.4f}_Kp{Kp:.3f}_Ki{Ki:.3f}_Kd{Kd:.3f}'
            print('Phase2 running', tag)

            # run
            config.PID_TRACE['enabled'] = True
            config.PID_TRACE['path'] = os.path.join(OUT_DIR, f'pid_trace_{tag}.csv')
            sim = simulation(port_name='Galveston', dt=DT, T=T_PHASE2, n_agents=1, load_enc=False,
                             test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
            sim.spawn()
            try:
                sim.ship.rudder_tau = float(tau)
            except Exception:
                pass
            sim.wind_fn = constant_crosswind_for_sim(sim, WIND_SPEED)
            sim.current_fn = lambda lon, lat, when: np.zeros((1, 2))

            start = time.time()
            sim.run()
            dur = time.time() - start

            if not os.path.exists(config.PID_TRACE['path']):
                print('Warning: missing trace for', tag)
                continue

            df = pd.read_csv(config.PID_TRACE['path'])
            df0 = df[df['agent'] == 0]
            max_err = float(df0['err_deg'].abs().max())
            mean_err = float(df0['err_deg'].abs().mean())
            max_raw = float(df0['raw_deg'].abs().max())
            max_rud = float(df0['rud_deg'].abs().max())
            sat_frac = float((df0['rud_deg'].abs() >= math.degrees(config.SHIP_PHYSICS['max_rudder']) - 1e-6).sum() / len(df0))

            prow = {
                'ndelta_mult': mult,
                'max_rudder_rate': mr,
                'rudder_tau': tau,
                'Kf_gain': kf,
                'Kp': Kp,
                'Ki': Ki,
                'Kd': Kd,
                'trace': config.PID_TRACE['path'],
                'max_err_deg': max_err,
                'mean_err_deg': mean_err,
                'max_raw_deg': max_raw,
                'max_rud_deg': max_rud,
                'sat_frac': sat_frac,
                'run_time_s': dur
            }
            phase2_rows.append(prow)
            pd.DataFrame(phase2_rows).to_csv(os.path.join(OUT_DIR, 'pid_refinement_summary.csv'), index=False)

    print('Phase 2 complete')


def restore():
    # restore original config
    config.SHIP_PHYSICS.update(orig_ship)
    config.CONTROLLER_GAINS.update(orig_ctrl)
    config.ADVANCED_CONTROLLER.update(orig_adv)


def main():
    try:
        phase1()
        phase2(top_k=3)
    finally:
        restore()
        print('Restored original config values')


if __name__ == '__main__':
    main()
