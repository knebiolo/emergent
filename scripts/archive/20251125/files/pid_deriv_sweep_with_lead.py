"""Small PID derivative sweep (Kd & deriv_tau) using lead_time=20s to see if D can damp oscillation.
Writes per-run traces to sweep_results/ and a summary.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

Kp_candidates = [0.35, 0.4, 0.45]
Kd_candidates = [0.25, 0.4, 0.6]
deriv_taus = [1.0, 2.0, 3.5]
lead_time = 20.0
release_deg = 5.0
DT = 0.5
T = 240.0

rows = []
for kp in Kp_candidates:
    for kd in Kd_candidates:
        for dtau in deriv_taus:
            name = f"kp{kp:.2f}_kd{kd:.2f}_dtau{dtau:.1f}"
            trace = os.path.join(OUT, f"pid_trace_sweep_{name}.csv")
            PID_TRACE['enabled'] = True
            PID_TRACE['path'] = trace
            sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                             test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
            sim.spawn()
            psi0 = float(sim.psi[0])
            wx = 5.0 * math.cos(psi0 + math.pi/2.)
            wy = 5.0 * math.sin(psi0 + math.pi/2.)
            sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
            sim.current_fn = lambda lon, lat, when: np.zeros((1,2))
            sim.ship.Kp = kp
            sim.ship.Ki = 0.0
            sim.ship.Kd = kd
            sim.ship.deriv_tau = dtau
            sim.tuning['Kp'] = kp
            sim.tuning['Ki'] = 0.0
            sim.tuning['Kd'] = kd
            sim.tuning['deriv_tau'] = dtau
            sim.tuning['lead_time'] = float(lead_time)
            sim.tuning['release_band_deg'] = float(release_deg)
            print('Running', name)
            sim.run()
            df = pd.read_csv(trace)
            df0 = df[df['agent']==0]
            err = df0['err_deg'].values
            rud = df0['rud_deg'].values
            mean_err = float(np.mean(np.abs(err)))
            std_err = float(np.std(err))
            pk2pk = float(np.max(err) - np.min(err))
            max_rud = float(np.max(np.abs(rud)))
            rows.append({'name': name, 'Kp': kp, 'Kd': kd, 'deriv_tau': dtau,
                         'mean_err_deg': mean_err, 'std_err_deg': std_err, 'pk2pk_err_deg': pk2pk,
                         'max_rud_deg': max_rud, 'trace': trace})
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'pid_deriv_sweep_lead20_summary.csv'), index=False)
            print('Completed', name)
print('All done')
