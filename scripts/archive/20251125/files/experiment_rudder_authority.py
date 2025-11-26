"""
Experiment: sweep Ndelta (yaw torque per rad) multipliers and summarize PID trace metrics.
Produces CSV traces in scripts/pid_trace_expt_<mult>.csv and prints a concise summary table.
"""
import os
import math
import shutil
import time
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS, PID_TRACE
from emergent.ship_abm.simulation_core import simulation

# Experiment parameters
multipliers = [1.0, 1.25, 1.5, 2.0, 3.0]
T = 60.0
dt = 0.1
n_agents = 1
port = 'Rosario Strait'

orig_Ndelta = float(SHIP_PHYSICS.get('Ndelta', 0.0))
max_rudder_deg = math.degrees(SHIP_PHYSICS.get('max_rudder', math.radians(20)))

results = []

for m in multipliers:
    tag = str(m).replace('.', 'p')
    trace_path = f'scripts/pid_trace_expt_{tag}.csv'
    # ensure PID_TRACE points to this file
    PID_TRACE['path'] = trace_path
    PID_TRACE['enabled'] = True

    # update Ndelta
    SHIP_PHYSICS['Ndelta'] = orig_Ndelta * m
    print(f"\n[EXPT] Running multiplier {m} (Ndelta={SHIP_PHYSICS['Ndelta']:.3e}) -> trace: {trace_path}")

    # remove old trace file if exists
    try:
        if os.path.exists(trace_path):
            os.remove(trace_path)
    except Exception:
        pass

    # construct & run sim
    sim = simulation(port_name=port, dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode='zigzag')
    sim.spawn()
    sim.run()

    # allow file flush
    time.sleep(0.1)

    # read CSV and compute metrics
    try:
        df = pd.read_csv(trace_path, header=None)
        # CSV writer in simulation writes header on first creation; handle either case
        if df.shape[1] == 10:
            df.columns = ['t','agent','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']
        else:
            # try reading with header row
            df = pd.read_csv(trace_path)
        # focus on agent 0
        df0 = df[df['agent'] == 0]
        max_err = df0['err_deg'].abs().max()
        max_raw = df0['raw_deg'].abs().max()
        mean_rud = df0['rud_deg'].mean()
        # saturation indicator: abs(rud_deg) >= max_rudder_deg - tiny
        sat_count = (df0['rud_deg'].abs() >= (max_rudder_deg - 1e-6)).sum()
        total = len(df0)
        sat_frac = sat_count / total if total>0 else 0.0
        # time to first sustained saturation (>= 1s contiguous) - optional
        results.append({'mult': m, 'max_err_deg': max_err, 'max_raw_deg': max_raw, 'mean_rud_deg': mean_rud, 'sat_frac': sat_frac, 'trace': trace_path})
        print(f"[EXPT] mult={m}: max_err={max_err:.2f}°, max_raw={max_raw:.2f}°, sat_frac={sat_frac:.2%}")
    except Exception as e:
        print(f"[EXPT] Failed to read trace for mult={m}: {e}")

# restore original Ndelta
SHIP_PHYSICS['Ndelta'] = orig_Ndelta

# print summary table
print('\nExperiment summary:')
for r in results:
    print(f"mult={r['mult']:>4}  max_err={r['max_err_deg']:6.2f}°  max_raw={r['max_raw_deg']:6.2f}°  sat_frac={r['sat_frac']:.2%}  trace={r['trace']}")

# done
print('\nAll done. Traces in scripts/*.csv')
