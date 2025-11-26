"""
Joint grid search over Ndelta, Kp multiplier, and Kd multiplier.
Runs short zig-zag tests and summarizes metrics to help pick a safe configuration.
"""
import os
import math
import time
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS, CONTROLLER_GAINS, PID_TRACE
from emergent.ship_abm.simulation_core import simulation

# grid
n_mults = [1.0, 1.25, 1.5]
kp_mults = [0.4, 0.6, 0.8]
kd_mults = [0.4, 0.6, 0.8]

# short test settings
T = 40.0
dt = 0.1
port = 'Rosario Strait'

orig_Ndelta = float(SHIP_PHYSICS.get('Ndelta', 0.0))
orig_Kp = float(CONTROLLER_GAINS.get('Kp', 0.6))
orig_Kd = float(CONTROLLER_GAINS.get('Kd', 0.5))

results = []

for n_m in n_mults:
    for kp_m in kp_mults:
        for kd_m in kd_mults:
            try:
                SHIP_PHYSICS['Ndelta'] = orig_Ndelta * n_m
                CONTROLLER_GAINS['Kp'] = orig_Kp * kp_m
                CONTROLLER_GAINS['Kd'] = orig_Kd * kd_m

                tag = f'n{int(n_m*100)}_kp{int(kp_m*100)}_kd{int(kd_m*100)}'
                trace = os.path.abspath(os.path.join('scripts', f'pid_trace_grid_{tag}.csv'))
                PID_TRACE['enabled'] = True
                PID_TRACE['path'] = trace
                if os.path.exists(trace):
                    try:
                        os.remove(trace)
                    except Exception:
                        pass
                print(f"[GRID] Nmult={n_m:.3f} Kp*={kp_m:.3f} Kd*={kd_m:.3f} -> {trace}")
                sim = simulation(port_name=port, dt=dt, T=T, n_agents=1, load_enc=False, test_mode='zigzag')
                sim.spawn()
                sim.run()
                time.sleep(0.05)
                # read trace
                df = pd.read_csv(trace)
                df0 = df[df['agent'] == 0]
                total = len(df0)
                max_err = float(df0['err_deg'].abs().max()) if total>0 else float('nan')
                max_raw = float(df0['raw_deg'].abs().max()) if total>0 else float('nan')
                mean_rud = float(df0['rud_deg'].mean()) if total>0 else float('nan')
                max_rud_deg = math.degrees(SHIP_PHYSICS.get('max_rudder', math.radians(20)))
                sat_frac = (df0['rud_deg'].abs() >= (max_rud_deg - 1e-6)).sum() / total if total>0 else 0.0
                results.append({'Nmult': n_m, 'kp_m': kp_m, 'kd_m': kd_m, 'max_err': max_err, 'max_raw': max_raw, 'mean_rud': mean_rud, 'sat_frac': sat_frac, 'trace': trace})
                print(f"[GRID] done: max_err={max_err:.2f}° max_raw={max_raw:.2f}° sat_frac={sat_frac:.2%}")
            except Exception as e:
                print('[GRID] run failed:', e)
                results.append({'Nmult': n_m, 'kp_m': kp_m, 'kd_m': kd_m, 'max_err': float('nan'), 'max_raw': float('nan'), 'mean_rud': float('nan'), 'sat_frac': 0.0, 'trace': trace})

# restore
SHIP_PHYSICS['Ndelta'] = orig_Ndelta
CONTROLLER_GAINS['Kp'] = orig_Kp
CONTROLLER_GAINS['Kd'] = orig_Kd

# sort by max_err then max_raw
results_sorted = sorted(results, key=lambda r: (math.isnan(r['max_err']), r['max_err'], r['max_raw']))

print('\nGrid search summary (best first):')
for r in results_sorted:
    print(f"Nmult={r['Nmult']:.2f} kp_m={r['kp_m']:.2f} kd_m={r['kd_m']:.2f}  max_err={r['max_err']:6.2f}°  max_raw={r['max_raw']:6.2f}°  sat_frac={r['sat_frac']:.2%}  trace={r['trace']}")

print('\nDone')
