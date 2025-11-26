import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emergent.ship_abm.simulation_core import simulation

T = 300.0
dt = 0.5
sim = simulation(port_name='Rosario Strait', dt=dt, T=T, n_agents=1, load_enc=False, test_mode='zigzag', zigzag_deg=10, zigzag_hold=40)
sim.spawn()
sim.run()

# The simulation records per-step positions in sim.history (dict per-agent) and timestamps in sim.t_history
hist = getattr(sim, 'history', None)
t_hist = getattr(sim, 't_history', None)

if hist is None or t_hist is None:
    # fallback: use final pos only (shouldn't happen in a properly recorded run)
    pos = getattr(sim, 'pos', None)
    if pos is None:
        raise RuntimeError('No trajectory recorded in simulation (sim.history and sim.t_history missing)')
    data = {'t': [0.0], 'x': [float(pos[0,0])], 'y': [float(pos[1,0]) ]}
else:
    # hist[0] is list of positions (each a length-2 array) recorded at each step for agent 0
    seq = hist.get(0, [])
    # ensure length matches t_hist (if last appended state was before final t, pad/truncate)
    n = min(len(seq), len(t_hist))
    xs = [float(seq[i][0]) for i in range(n)]
    ys = [float(seq[i][1]) for i in range(n)]
    ts = [float(t_hist[i]) for i in range(n)]
    data = {'t': ts, 'x': xs, 'y': ys}

os.makedirs('scripts', exist_ok=True)
csv_path = 'scripts/planimetric_track.csv'
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)
print('Wrote', csv_path)

# Plot planimetric track
plt.figure(figsize=(6,6))
plt.plot(df['x'], df['y'], '-k')
plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='g', label='start')
plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='r', label='end')
plt.axis('equal')
plt.title('Planimetric track (UTM)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
os.makedirs('plots', exist_ok=True)
png = 'plots/planimetric_track.png'
plt.savefig(png)
print('Wrote', png)
