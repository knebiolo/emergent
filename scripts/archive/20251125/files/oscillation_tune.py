"""Targeted oscillation tuning: run a small set of controller candidates and compute oscillation metrics.
Saves results to sweep_results/oscillation_tune_summary.csv and per-run pid traces in sweep_results/.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, CONTROLLER_GAINS, ADVANCED_CONTROLLER, SHIP_PHYSICS

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1
T = 400.0
WIND_SPEED = 5.0

candidates = [
    {
        'name': 'lowKp_Kp0.4_Kd0.25_dtau1.0',
        'Kp': 0.4, 'Ki': 0.0, 'Kd': 0.25, 'deriv_tau': 1.0, 'release_band_deg': ADVANCED_CONTROLLER.get('release_band_deg',3.0)
    },
    {
        'name': 'keepKp_lowerKd_dtau2.0',
        'Kp': 0.5, 'Ki': 0.0, 'Kd': 0.2, 'deriv_tau': 2.0, 'release_band_deg': ADVANCED_CONTROLLER.get('release_band_deg',3.0)
    },
    {
        'name': 'midKp_Kd_dtau1p5_release5',
        'Kp': 0.45, 'Ki': 0.0, 'Kd': 0.25, 'deriv_tau': 1.5, 'release_band_deg': 5.0
    }
]

rows = []

for c in candidates:
    trace = os.path.join(OUT_DIR, f"pid_trace_osc_{c['name']}.csv")
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = trace

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
    sim.spawn()
    # set wind
    psi0 = float(sim.psi[0])
    wx = WIND_SPEED * math.cos(psi0 + math.pi/2.)
    wy = WIND_SPEED * math.sin(psi0 + math.pi/2.)
    sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
    sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

    # apply gains on ship
    try:
        sim.ship.Kp = c['Kp']
        sim.ship.Ki = c['Ki']
        sim.ship.Kd = c['Kd']
        # set sim.ship attributes (keeps ship object fields consistent)
        sim.ship.deriv_tau = c['deriv_tau']
        # also set release band on ship as degrees->radians consistent with ship fields
        sim.ship.release_band_deg = math.radians(c.get('release_band_deg', ADVANCED_CONTROLLER.get('release_band_deg', 3.0)))
        # IMPORTANT: the controller uses simulation.tuning at runtime, so update it too
        try:
            sim.tuning['Kp'] = float(c['Kp'])
            sim.tuning['Ki'] = float(c['Ki'])
            sim.tuning['Kd'] = float(c['Kd'])
            sim.tuning['deriv_tau'] = float(c['deriv_tau'])
            sim.tuning['release_band_deg'] = float(c.get('release_band_deg', ADVANCED_CONTROLLER.get('release_band_deg', 3.0)))
        except Exception:
            pass
    except Exception:
        # fallback to globals
        CONTROLLER_GAINS['Kp'] = c['Kp']
        CONTROLLER_GAINS['Ki'] = c['Ki']
        CONTROLLER_GAINS['Kd'] = c['Kd']
        ADVANCED_CONTROLLER['lead_time'] = ADVANCED_CONTROLLER.get('lead_time',20.0)
        ADVANCED_CONTROLLER['release_band_deg'] = c.get('release_band_deg', ADVANCED_CONTROLLER.get('release_band_deg',3.0))

    print('Running candidate', c['name'])
    sim.run()

    # read trace
    df = pd.read_csv(trace)
    df0 = df[df['agent'] == 0]
    t = df0['t'].values
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values

    # metrics
    mean_err = np.mean(np.abs(err))
    std_err = np.std(err)
    pk2pk_err = np.max(err) - np.min(err)
    max_rud = np.max(np.abs(rud))
    sat_frac = float((np.abs(rud) >= math.degrees(SHIP_PHYSICS['max_rudder']) - 1e-6).sum()) / len(rud)

    # dominant frequency via FFT
    try:
        n = len(err)
        # detrend
        y = err - np.mean(err)
        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(n, d=(t[1]-t[0]) if len(t)>1 else DT)
        idx = np.argmax(np.abs(yf[1:])) + 1
        dom_freq = float(xf[idx])
        dom_period = 1.0/dom_freq if dom_freq>0 else float('nan')
        dom_amp = float(np.abs(yf[idx])/n)
    except Exception:
        dom_freq = float('nan')
        dom_period = float('nan')
        dom_amp = float('nan')

    rows.append({
        'name': c['name'], 'Kp': c['Kp'], 'Ki': c['Ki'], 'Kd': c['Kd'],
        'deriv_tau': c['deriv_tau'], 'release_band_deg': c.get('release_band_deg', ADVANCED_CONTROLLER.get('release_band_deg',3.0)),
        'mean_err_deg': mean_err, 'std_err_deg': std_err, 'pk2pk_err_deg': pk2pk_err,
        'max_rud_deg': max_rud, 'sat_frac': sat_frac, 'dom_freq_hz': dom_freq, 'dom_period_s': dom_period, 'dom_amp': dom_amp,
        'trace': trace
    })

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'oscillation_tune_summary.csv'), index=False)
    print('Completed', c['name'])

print('All candidates done. Summary at', os.path.join(OUT_DIR, 'oscillation_tune_summary.csv'))
