import numpy as np
import pandas as pd
from emergent.ship_abm.simulation_core import simulation

T = 300.0
dt = 0.5
sim = simulation(port_name='Rosario Strait', dt=dt, T=T, n_agents=1, load_enc=False, test_mode='zigzag', zigzag_deg=10, zigzag_hold=40)
sim.spawn()
# We'll record wind magnitudes each step by wrapping the wind_fn
wind_samples = []

# run manually stepping in run() loop is not accessible externally; re-run sim.run and afterwards inspect sim.log_lines
sim.run()
# sim.log_lines includes strings of the form 'Wind=... Curr=... Tide=...'
lines = getattr(sim, 'log_lines', [])
print('Sample log lines (last 5):')
for l in lines[:5]:
    print(l)

# try to parse wind entries from sim.log_lines
winds = []
for l in lines:
    # look for prefix 'Wind='
    if 'Wind=' in l:
        try:
            part = l.split()[0]
            # Format: Wind=%.1f
            wstr = part.split('=')[1]
            winds.append(float(wstr))
        except Exception:
            pass

if winds:
    import matplotlib.pyplot as plt
    import os
    os.makedirs('plots', exist_ok=True)
    pd.DataFrame({'wind': winds}).to_csv('scripts/wind_samples.csv', index=False)
    plt.plot(winds)
    plt.title('Logged wind magnitudes (sample)')
    plt.savefig('plots/wind_samples.png')
    print('Wrote scripts/wind_samples.csv and plots/wind_samples.png')
else:
    print('No wind entries found in sim.log_lines')