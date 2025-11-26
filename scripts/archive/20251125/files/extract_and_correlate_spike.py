"""Extract sim-history window around a spike and correlate with PID CSV.

Creates two artifacts in scripts/:
 - pid_trace_repro_t{T}.csv (from repro run)
 - simhist_repro_t{T}.npz   (contains t, psi, hd_cmd, pos)
 - pid_trace_repro_t{T}_correlated.csv (correlation output)

Usage: python scripts/extract_and_correlate_spike.py [spike_time]
"""
import os
import sys
import math
import numpy as np
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE

SPIKE_T = float(sys.argv[1]) if 'sys' in globals() and len(sys.argv) > 1 else 478.5
PRE = 5.0
POST = 5.0
DT = 0.5
N_AGENTS = 1
WAYPOINTS = [[(0.0, 0.0), (500.0, 0.0)]]

# simple constant env functions to reproduce the case
W_SPEED = 8.0
C_SPEED = 0.5

def constant_wind_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-W_SPEED]]), (1, N_AGENTS))

def constant_current_fn(lon, lat, now):
    return np.tile(np.array([[0.0], [-C_SPEED]]), (1, N_AGENTS))

# prepare paths
out_csv = os.path.join('scripts', f'pid_trace_repro_t{SPIKE_T:.1f}.csv')
simnpz = os.path.join('scripts', f'simhist_repro_t{SPIKE_T:.1f}.npz')
correlated = os.path.join('scripts', f'pid_trace_repro_t{SPIKE_T:.1f}_correlated.csv')

# ensure PID_TRACE writes to the expected CSV
PID_TRACE['enabled'] = True
PID_TRACE['path'] = out_csv
if os.path.exists(out_csv):
    os.remove(out_csv)

# run sim
steps = int(math.ceil((SPIKE_T + POST) / DT))
T = steps * DT
print(f"Running repro sim T={T}s dt={DT} spike_t={SPIKE_T}")

sim = simulation(port_name='Galveston', dt=DT, T=T, n_agents=N_AGENTS, load_enc=False, test_mode=None)
sim.wind_fn = constant_wind_fn
sim.current_fn = constant_current_fn
sim.waypoints = WAYPOINTS
sim.spawn()
sim.run()

# save sim history
np.savez_compressed(simnpz, t=np.array(sim.t_history), psi=np.array(sim.psi_history), hd_cmd=np.array(sim.hd_cmd_history), pos=np.array(sim.pos_history) if hasattr(sim, 'pos_history') else np.array(sim.pos))
print('Wrote simhistory to', simnpz)

# call correlate script
import subprocess
ret = subprocess.run([sys.executable, 'scripts/correlate_pid_with_simhist.py', out_csv, simnpz, '160.0', correlated], capture_output=True, text=True)
print(ret.stdout)
if ret.stderr:
    print('STDERR:', ret.stderr)

print('Done')
