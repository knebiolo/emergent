"""Re-run a short sim around a spike and print PID CSV and sim arrays for inspection.

Usage: python scripts/repro_spike_debug.py

This enables PID_TRACE to a local file and runs a short simulation for the chosen
spike time (configured inside). It then prints the CSV contents and the sim's
psi/hd_cmd/time arrays for comparison.
"""
import os
import math
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

# choose spike time from earlier analysis
SPIKE_T = 478.5
PRE = 3.0
POST = 3.0
DT = 0.5
N_AGENTS = 1
WAYPOINTS = [[(0.0, 0.0), (500.0, 0.0)]]

# constant env functions
W_SPEED = 8.0
C_SPEED = 0.5

def constant_wind_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-W_SPEED]]), (1, N_AGENTS))

def constant_current_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-C_SPEED]]), (1, N_AGENTS))

# configure PID_TRACE to local output
out_csv = os.path.join(os.getcwd(), 'scripts', f'pid_trace_repro_t{SPIKE_T:.1f}.csv')
PID_TRACE['enabled'] = True
PID_TRACE['path'] = out_csv
if os.path.exists(out_csv):
    os.remove(out_csv)

# build simulation
t_start = max(0.0, SPIKE_T - PRE)
steps = int(math.ceil((SPIKE_T + POST) / DT))
T = steps * DT
print(f"Running repro sim T={T}s dt={DT} spike_t={SPIKE_T}")

sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, test_mode=None)
sim.wind_fn = constant_wind_fn
sim.current_fn = constant_current_fn
sim.waypoints = WAYPOINTS
sim.spawn()
sim.run()

print('\n=== PID CSV content ===')
with open(out_csv, 'r') as fh:
    for i, line in enumerate(fh):
        print(line.strip())
        if i > 80:
            break

print('\n=== Sim arrays close to spike ===')
times = np.array(sim.t_history)
psi = np.array(sim.psi_history)
hd_cmd = np.array(sim.hd_cmd_history)

# show neighborhood around SPIKE_T
if len(times) == 0:
    print('No times recorded')
else:
    idx = int(np.argmin(np.abs(times - SPIKE_T)))
    lo = max(0, idx-6)
    hi = min(len(times)-1, idx+6)
    print('times_idx, time, psi_deg, hd_cmd_deg')
    for k in range(lo, hi+1):
        print(k, f"{times[k]:.1f}", f"{np.degrees(psi[k]):+.3f}", f"{np.degrees(hd_cmd[k]):+.3f}")

print('\nDone')
