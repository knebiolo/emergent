from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE
import os

# configure trace path
PID_TRACE['path'] = 'scripts/pid_trace_autodetect.csv'
PID_TRACE['enabled'] = True

# create simulation and run headless
sim = simulation(port_name='Rosario Strait', dt=0.1, T=60, n_agents=1, load_enc=False, test_mode='zigzag')
# initialize spawn (create minimal waypoints/state for headless run)
sim.spawn()
# run the simulation loop
sim.run()
print('Done')
