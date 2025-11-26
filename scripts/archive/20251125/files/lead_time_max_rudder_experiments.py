"""Run targeted experiments varying lead_time and max_rudder while using the current best PID.
Writes per-run pid traces to sweep_results/ and an aggregated CSV.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, CONTROLLER_GAINS, ADVANCED_CONTROLLER, SHIP_PHYSICS

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

# best candidate from earlier: lowKp
Kp = 0.4
Ki = 0.0
Kd = 0.25

# experiments: tuples of (max_rudder_deg, lead_time_s, release_band_deg)
exps = [
    (14.0, 20.0, 5.0),
    (25.0, 20.0, 5.0),
    (35.0, 20.0, 5.0)
]

DT = 0.1
T = 300.0
WIND_SPEED = 5.0

rows = []
for mr_deg, lead_time, release_deg in exps:
    name = f"lead{int(lead_time)}_mr{int(mr_deg)}_rel{int(release_deg)}"
    trace = os.path.join(OUT, f"pid_trace_{name}.csv")
    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = trace

    sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
    sim.spawn()
    psi0 = float(sim.psi[0])
    wx = WIND_SPEED * math.cos(psi0 + math.pi/2.)
    wy = WIND_SPEED * math.sin(psi0 + math.pi/2.)
    sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
    sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

    # apply PID and tuning
    sim.ship.Kp = Kp
    sim.ship.Ki = Ki
    sim.ship.Kd = Kd
    sim.ship.deriv_tau = 1.0
    sim.ship.release_band_deg = math.radians(release_deg)
    # ensure simulation tuning dict is updated
    sim.tuning['Kp'] = Kp
    sim.tuning['Ki'] = Ki
    sim.tuning['Kd'] = Kd
    sim.tuning['lead_time'] = float(lead_time)
    sim.tuning['release_band_deg'] = float(release_deg)

    # override max rudder on the ship
    sim.ship.max_rudder = math.radians(mr_deg)

    print('Running', name)
    sim.run()

    # read trace and compute metrics
    df = pd.read_csv(trace)
    df0 = df[df['agent']==0]
    t = df0['t'].values
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values
    mean_err = float(np.mean(np.abs(err)))
    std_err = float(np.std(err))
    pk2pk = float(np.max(err) - np.min(err))
    max_rud = float(np.max(np.abs(rud)))
    sat_frac = float((np.abs(rud) >= (mr_deg - 1e-6)).sum()) / len(rud)

    rows.append({'name': name, 'max_rudder_deg': mr_deg, 'lead_time_s': lead_time, 'release_band_deg': release_deg,
                 'mean_err_deg': mean_err, 'std_err_deg': std_err, 'pk2pk_err_deg': pk2pk,
                 'max_rud_deg': max_rud, 'sat_frac': sat_frac, 'trace': trace})

    # write interim CSV
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'lead_time_max_rudder_summary.csv'), index=False)
    print('Completed', name)

print('All experiments done. Summary:', os.path.join(OUT, 'lead_time_max_rudder_summary.csv'))
