"""
Headless two-agent run in Rosario Strait (parallel waypoints) for verification.
Writes PID trace and basic summary (collisions/allisions).
"""
import os
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm import config
from emergent.ship_abm.config import PID_TRACE

# tuning & sim settings
config.SHIP_AERO_DEFAULTS['wind_force_scale'] = 0.35
config.CONTROLLER_GAINS['Kd'] = 3.0
config.ADVANCED_CONTROLLER['deriv_tau'] = 1.0
config.ADVANCED_CONTROLLER['trim_band_deg'] = 2.0

PID_TRACE['enabled'] = True
PID_TRACE['path'] = os.path.join('traces', 'rosario_2_agent_pid_trace.csv')

T = 600.0
dt = 0.5

if __name__ == '__main__':
    sim = simulation(port_name='Rosario Strait', dt=dt, T=T, n_agents=2, load_enc=False, test_mode=None)
    # Create simple parallel waypoints across the strait in meters (UTM-like local coords)
    # We'll place two straight routes separated in y so they are parallel and should not collide.
    # Waypoints expressed in meters relative to spawn: start at x=-2000 -> x=2000
    wps = []
    wps.append([(-2000.0, -200.0), (2000.0, -200.0)])
    wps.append([(-2000.0,  200.0), (2000.0,  200.0)])
    sim.waypoints = wps
    sim.spawn()
    print(f"Starting headless Rosario Strait sim with 2 agents for T={T}s, dt={dt}s")
    sim.run()

    # Summarize
    print('Simulation complete. Summary:')
    print('  Collision events:', sim.collision_events)
    print('  Allision events:', sim.allision_events)
    print('  Trace path:', PID_TRACE['path'])
