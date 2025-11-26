"""Test reducing ship.rudder_tau (actuator bandwidth) to see effect on oscillation.
Runs a single test with current best PID and lead_time=20s; writes trace and summary.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

Kp = 0.4
Ki = 0.0
Kd = 0.25
lead_time = 20.0
release_deg = 5.0

for tau in [1.0, 0.6, 0.4]:
    name = f"rudder_tau_{tau:.2f}_lead{int(lead_time)}"
    trace = os.path.join(OUT, f"pid_trace_{name}.csv")
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = trace
    sim = simulation(port_name='Galveston', dt=0.1, T=300.0, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
    sim.spawn()
    psi0 = float(sim.psi[0])
    wx = 5.0 * math.cos(psi0 + math.pi/2.)
    wy = 5.0 * math.sin(psi0 + math.pi/2.)
    sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
    sim.current_fn = lambda lon, lat, when: np.zeros((1,2))
    # set pid and tuning
    sim.ship.Kp = Kp
    sim.ship.Ki = Ki
    sim.ship.Kd = Kd
    sim.ship.deriv_tau = 1.0
    sim.ship.release_band_deg = math.radians(release_deg)
    sim.tuning['Kp'] = Kp
    sim.tuning['Ki'] = Ki
    sim.tuning['Kd'] = Kd
    sim.tuning['lead_time'] = float(lead_time)
    sim.tuning['release_band_deg'] = float(release_deg)
    # set rudder tau
    sim.ship.rudder_tau = float(tau)
    print('Running', name)
    sim.run()
    # compute metrics
    df = pd.read_csv(trace)
    df0 = df[df['agent']==0]
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values
    mean_err = float(np.mean(np.abs(err)))
    max_rud = float(np.max(np.abs(rud)))
    pd.DataFrame([{'name': name, 'rudder_tau': tau, 'mean_err_deg': mean_err, 'max_rud_deg': max_rud, 'trace': trace}]).to_csv(os.path.join(OUT, f'rudder_tau_{tau:.2f}_summary.csv'), index=False)
    print('Completed', name)
print('All done')
