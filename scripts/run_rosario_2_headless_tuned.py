"""
Conservative tuned headless two-agent run in Rosario Strait.
Reduces Kp/Ki/Kf and rudder limits to test whether spinning is eliminated.
Writes PID trace to traces/rosario_2_agent_pid_trace_tuned.csv
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config
from emergent.ship_abm.config import PID_TRACE

# conservative tuning
config.SHIP_AERO_DEFAULTS['wind_force_scale'] = 0.35
config.CONTROLLER_GAINS['Kp'] = 0.3
config.CONTROLLER_GAINS['Ki'] = 0.02
config.CONTROLLER_GAINS['Kd'] = 0.12
config.ADVANCED_CONTROLLER['Kf_gain'] = 0.001

# reduce rudder authority and rate
config.SHIP_PHYSICS['max_rudder'] = np.radians(15.0)
config.SHIP_PHYSICS['max_rudder_rate'] = 0.12

PID_TRACE['enabled'] = True
PID_TRACE['path'] = os.path.join('traces', 'rosario_2_agent_pid_trace_tuned.csv')

T = 600.0
dt = 0.5

if __name__ == '__main__':
    sim = simulation(port_name='Rosario Strait', dt=dt, T=T, n_agents=2, load_enc=False, test_mode=None)
    wps = []
    wps.append([(-2000.0, -200.0), (2000.0, -200.0)])
    wps.append([(-2000.0,  200.0), (2000.0,  200.0)])
    sim.waypoints = wps
    sim.spawn()
    print(f"Starting tuned headless Rosario sim with 2 agents for T={T}s, dt={dt}s")
    sim.run()
    print('Simulation complete. Trace path:', PID_TRACE['path'])
