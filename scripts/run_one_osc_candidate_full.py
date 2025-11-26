"""Run the specific oscillation candidate 'keepKp_lowerKd_dtau2.0' full-length (T=400s) and capture output.
"""
import os, math, numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config

OUT_DIR = 'sweep_results'
os.makedirs(OUT_DIR, exist_ok=True)

DT = 0.1
T = 400.0
WIND_SPEED = 5.0

c = {
    'name': 'keepKp_lowerKd_dtau2.0_full',
    'Kp': 0.5, 'Ki': 0.0, 'Kd': 0.2, 'deriv_tau': 2.0, 'release_band_deg': 3.0
}

trace = os.path.join(OUT_DIR, f"pid_trace_osc_{c['name']}.csv")
config.PID_TRACE['enabled'] = True
config.PID_TRACE['path'] = trace

sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=1, load_enc=False, test_mode='zigzag', zigzag_deg=15, zigzag_hold=30)
sim.spawn()
psi0 = float(sim.psi[0])
wx = WIND_SPEED * math.cos(psi0 + math.pi/2.)
wy = WIND_SPEED * math.sin(psi0 + math.pi/2.)
sim.wind_fn = lambda lon, lat, when: np.tile(np.array([[wx, wy]]), (1,1))
sim.current_fn = lambda lon, lat, when: np.zeros((1,2))

# apply gains
sim.ship.Kp = c['Kp']
sim.ship.Ki = c['Ki']
sim.ship.Kd = c['Kd']
sim.ship.deriv_tau = c['deriv_tau']

print('Running candidate', c['name'], 'T=', T)
sim.run()
print('Run complete; trace at', trace)
