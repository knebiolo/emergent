"""
Run a long headless zigzag test with conservative PID tuning and write a PID trace.
"""
from emergent.ship_abm import config
from emergent.ship_abm.config import PID_TRACE
from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics
import numpy as np

# Conservative tuning for long run
config.SHIP_AERO_DEFAULTS['wind_force_scale'] = 0.35
config.CONTROLLER_GAINS['Kd'] = 3.0
config.ADVANCED_CONTROLLER['trim_band_deg'] = 2.0
config.ADVANCED_CONTROLLER['lead_time'] = 5.0
config.ADVANCED_CONTROLLER['deriv_tau'] = 1.0

PID_TRACE['enabled'] = True
PID_TRACE['path'] = 'pid_trace_zigzag_long.csv'

if __name__ == '__main__':
    sim = simulation(port_name='Baltimore', dt=0.5, T=360.0, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=10, zigzag_hold=60)
    sim.spawn()
    total_T = sim.steps * sim.dt
    print(f"Running long zigzag (T={total_T}s, dt={sim.dt}s) -> {PID_TRACE['path']}")
    sim.run()
    t = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd = np.array(sim.hd_cmd_history)
    metrics = compute_zigzag_metrics(t, psi, hd, tol=3.0)
    print('Zigzag metrics:')
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print('Trace file:', PID_TRACE['path'])
