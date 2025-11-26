"""
Grid search over Kp and Kd multipliers with Ndelta fixed to 1.5x.
Writes per-run traces to scripts/pid_trace_tune_kp<k>_kd<k>.csv and prints a summary table.
"""
import os
import time
import math
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS, CONTROLLER_GAINS, PID_TRACE
from emergent.ship_abm.simulation_core import simulation

orig_Ndelta = float(SHIP_PHYSICS.get('Ndelta', 0.0))
orig_Kp = float(CONTROLLER_GAINS.get('Kp', 0.6))
orig_Kd = float(CONTROLLER_GAINS.get('Kd', 0.5))

kp_mults = [0.6, 0.8, 1.0]
kd_mults = [0.6, 0.8, 1.0]

T = 40.0
dt = 0.1
n_agents = 1
port = 'Rosario Strait'

results = []

# set Ndelta to 1.5x for the grid search
SHIP_PHYSICS['Ndelta'] = orig_Ndelta * 1.5

for kp_m in kp_mults:
    for kd_m in kd_mults:
        CONTROLLER_GAINS['Kp'] = orig_Kp * kp_m
        CONTROLLER_GAINS['Kd'] = orig_Kd * kd_m
        tag = f'kp{int(kp_m*100)}_kd{int(kd_m*100)}'
        trace_path = f'scripts/pid_trace_tune_{tag}.csv'
        PID_TRACE['path'] = trace_path
        PID_TRACE['enabled'] = True
        try:
            if os.path.exists(trace_path):
                os.remove(trace_path)
        except Exception:
            pass
        print(f"[TUNE] kp={CONTROLLER_GAINS['Kp']:.3f} kd={CONTROLLER_GAINS['Kd']:.3f} -> {trace_path}")
        sim = simulation(port_name=port, dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode='zigzag')
        sim.spawn()
        sim.run()
        time.sleep(0.1)
        # read trace
        try:
            df = pd.read_csv(trace_path)
            df0 = df[df['agent'] == 0]
            total = len(df0)
            max_err = df0['err_deg'].abs().max()
            max_raw = df0['raw_deg'].abs().max()
            sat_frac = (df0['rud_deg'].abs() >= (math.degrees(SHIP_PHYSICS.get('max_rudder', math.radians(20))) - 1e-6)).sum() / total if total>0 else 0.0
            results.append({'kp_m': kp_m, 'kd_m': kd_m, 'max_err': max_err, 'max_raw': max_raw, 'sat_frac': sat_frac, 'trace': trace_path})
            print(f"[TUNE] done: max_err={max_err:.2f}° max_raw={max_raw:.2f}° sat_frac={sat_frac:.2%}")
        except Exception as e:
            print('[TUNE] failed to read trace:', e)
            results.append({'kp_m': kp_m, 'kd_m': kd_m, 'max_err': math.nan, 'max_raw': math.nan, 'sat_frac': 0.0, 'trace': trace_path})

# restore
SHIP_PHYSICS['Ndelta'] = orig_Ndelta
CONTROLLER_GAINS['Kp'] = orig_Kp
CONTROLLER_GAINS['Kd'] = orig_Kd

print('\nGrid search results:')
for r in results:
    print(f"kp_m={r['kp_m']:>4} kd_m={r['kd_m']:>4}  max_err={r['max_err']:6.2f}°  max_raw={r['max_raw']:6.2f}°  sat_frac={r['sat_frac']:.2%}  trace={r['trace']}")

print('\nDone')
