"""Combined plant × controller sweep

Tests combinations of rudder actuator bandwidth (`rudder_tau`) and maximum rudder
angle (`max_rudder` in degrees) using the currently-applied controller gains.

Writes per-run PID trace CSVs into `sweep_results/` and a compact summary CSV
`sweep_results/combined_plant_controller_summary.csv`.
"""
import os
import math
import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE, SHIP_PHYSICS

OUT = 'sweep_results'
os.makedirs(OUT, exist_ok=True)

# Parameter grid
rudder_taus = [1.0, 0.6, 0.4]
max_rud_deg = [14, 25, 35]

# Simulation settings (zigzag stress test)
DT = 0.1
T = 300.0
zigdeg = 15
zighold = 30
wind_speed = 5.0

rows = []
for tau in rudder_taus:
    for mr in max_rud_deg:
        name = f"rtau{tau:.2f}_mr{mr}deg"
        trace = os.path.join(OUT, f"pid_trace_comb_{name}.csv")
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

        # Apply plant-level parameters on the ship object
        sim.ship.rudder_tau = float(tau)
        sim.ship.max_rudder = float(math.radians(mr))

        # Ensure controller tuning dictionary matches expected values (use current config as base)
        sim.tuning['Kp'] = float(sim.tuning.get('Kp', 0.0))
        sim.tuning['Ki'] = float(sim.tuning.get('Ki', 0.0))
        sim.tuning['Kd'] = float(sim.tuning.get('Kd', 0.0))
        sim.tuning['deriv_tau'] = float(sim.tuning.get('deriv_tau', 1.0))
        sim.tuning['lead_time'] = float(sim.tuning.get('lead_time', 20.0))
        sim.tuning['release_band_deg'] = float(sim.tuning.get('release_band_deg', 5.0))

        print('Running', name, f'(tau={tau}, max_rud={mr}°)')
        sim.run()

        # read trace and compute metrics
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
        sat_frac = float((df0['rud_deg'].abs() >= math.degrees(sim.ship.max_rudder) - 1e-6).sum() / len(df0))

        rows.append({'name': name, 'rudder_tau': tau, 'max_rudder_deg': mr,
                     'mean_err_deg': mean_err, 'std_err_deg': std_err,
                     'pk2pk_err_deg': pk2pk, 'max_rud_deg': max_rud_applied,
                     'sat_frac': sat_frac, 'trace': trace})

        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'combined_plant_controller_summary.csv'), index=False)
        print('Completed', name, 'mean_err_deg=', mean_err)

print('All done')
