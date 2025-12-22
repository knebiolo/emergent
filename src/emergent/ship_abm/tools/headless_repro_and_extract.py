import os
import csv
from emergent.ship_abm.simulation_core import simulation

# Ensure deep debug writer is enabled
os.environ['EMERGENT_PID_DEEP_DEBUG'] = '1'

# Headless simulation parameters
port = 'Seattle'
dt = 0.1
T = 350
n_agents = 2

sim = simulation(port_name=port, dt=dt, T=T, n_agents=n_agents, load_enc=False, verbose=False)
# Provide simple straight routes for two agents to allow spawn()
sim.waypoints = [ [(0.0, 0.0), (1000.0, 0.0)], [(0.0, 50.0), (1000.0, 50.0)] ]

# Initialize and run
sim.spawn()
sim.run()
print('SIM_RUN_COMPLETE')

# Helper to filter CSV rows in time window for agent 1
logs_dir = os.path.join(os.getcwd(), 'logs')
files = [ 'pid_mismatch_debug.csv', 'pid_deep_debug.csv', 'colregs_runtime_debug.csv' ]
window_lo = 220.0
window_hi = 285.0

for fname in files:
    path = os.path.join(logs_dir, fname)
    print('\n---', fname, '---')
    if not os.path.exists(path):
        print('MISSING', path)
        continue
    try:
        with open(path, 'r', newline='') as fh:
            reader = csv.reader(fh)
            hdr = next(reader, None)
            print('HEADER:', hdr)
            count = 0
            for row in reader:
                try:
                    t = float(row[0])
                except Exception:
                    continue
                if t < window_lo or t > window_hi:
                    continue
                # focus on agent 1 rows (agent index '1')
                try:
                    a = int(row[1])
                except Exception:
                    a = None
                if a is not None and a != 1:
                    continue
                print(row)
                count += 1
                if count >= 200:
                    print('...truncated after 200 rows')
                    break
            if count == 0:
                print('No rows found in window')
    except Exception as e:
        print('Error reading', path, e)
print('\nEXTRACTION_DONE')
