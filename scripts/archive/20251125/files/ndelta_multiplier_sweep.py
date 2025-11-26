"""Sweep Ndelta (rudder yaw torque) multipliers and rudder_tau.

For each combo, set sim.ship.Ndelta = base * multiplier and sim.ship.rudder_tau = tau,
run the zigzag crosswind test, and record summary metrics.

Writes per-run PID traces into `sweep_results/` and a summary CSV
`sweep_results/ndelta_multiplier_summary.csv`.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

# grid
ndelta_mults = [1.0, 1.25, 1.5, 2.0]
rudder_taus = [1.0, 0.6]

# sim settings
DT = 0.1
T = 300.0
zigdeg = 15
zighold = 30
wind_speed = 5.0

rows = []
for mult in ndelta_mults:
    for tau in rudder_taus:
        name = f"nd{mult:.2f}_rtau{tau:.2f}"
        trace = os.path.join(OUT, f"pid_trace_nd_{name}.csv")
        PID_TRACE['enabled'] = True
        PID_TRACE['path'] = trace

        sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                         test_mode='zigzag', zigzag_deg=zigdeg, zigzag_hold=zighold)
        sim.spawn()

        # constant perpendicular crosswind (starboard)
        psi0 = float(sim.psi[0])
        cross_theta = psi0 + math.pi/2.0
        wx = wind_speed * math.cos(cross_theta)
        wy = wind_speed * math.sin(cross_theta)
        sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
        sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

        # apply multipliers and plant params
        base_Ndelta = float(sim.ship.Ndelta)
        sim.ship.Ndelta = base_Ndelta * float(mult)
        sim.ship.rudder_tau = float(tau)

        # ensure controller tuning dict remains consistent
        sim.tuning['Kp'] = float(sim.tuning.get('Kp', sim.ship.Kp))
        sim.tuning['Ki'] = float(sim.tuning.get('Ki', sim.ship.Ki))
        sim.tuning['Kd'] = float(sim.tuning.get('Kd', sim.ship.Kd))
        sim.tuning['deriv_tau'] = float(sim.tuning.get('deriv_tau', sim.ship.deriv_tau))
        sim.tuning['lead_time'] = float(sim.tuning.get('lead_time', sim.ship.lead_time))
        sim.tuning['release_band_deg'] = float(sim.tuning.get('release_band_deg', math.degrees(sim.ship.release_band_deg)))

        print('Running', name, f'(ndelta_mult={mult}, rudder_tau={tau})')
        sim.run()

        # read trace
        try:
            df = pd.read_csv(trace)
        except Exception:
            print('Missing trace for', name)
            continue
        df0 = df[df['agent'] == 0]
        err = df0['err_deg'].values
        rud = df0['rud_deg'].values
        mean_err = float(np.mean(np.abs(err)))
        std_err = float(np.std(err))
        pk2pk = float(np.max(err) - np.min(err))
        max_rud_applied = float(np.max(np.abs(rud)))
        sat_frac = float((df0['rud_deg'].abs() >= (math.degrees(sim.ship.max_rudder) - 1e-6)).sum() / len(df0))

        rows.append({'name': name, 'ndelta_mult': mult, 'rudder_tau': tau,
                     'mean_err_deg': mean_err, 'std_err_deg': std_err,
                     'pk2pk_err_deg': pk2pk, 'max_rud_deg': max_rud_applied,
                     'sat_frac': sat_frac, 'trace': trace})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'ndelta_multiplier_summary.csv'), index=False)
        print('Completed', name, 'mean_err_deg=', mean_err)

print('All done')
