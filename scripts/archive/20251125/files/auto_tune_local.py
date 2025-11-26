"""
Run a small Kp/Kd grid (local to current config) using zig-zag test_mode and summarize PID trace metrics.
This script intentionally DOES NOT change SHIP_PHYSICS['Ndelta'] beyond what's in `config.py` (so it uses the tuned value there).
"""
import os
import math
import time
import pandas as pd
from emergent.ship_abm.config import CONTROLLER_GAINS, PID_TRACE, SHIP_PHYSICS
from emergent.ship_abm.simulation_core import simulation

kp_candidates = [CONTROLLER_GAINS['Kp'], CONTROLLER_GAINS['Kp']*1.25, CONTROLLER_GAINS['Kp']*1.5]
kd_candidates = [CONTROLLER_GAINS['Kd'], CONTROLLER_GAINS['Kd']*1.2]

T = 120.0
dt = 0.5
port = 'Rosario Strait'

results = []
orig_kp = CONTROLLER_GAINS['Kp']
orig_kd = CONTROLLER_GAINS['Kd']

for kp in kp_candidates:
    for kd in kd_candidates:
        CONTROLLER_GAINS['Kp'] = kp
        CONTROLLER_GAINS['Kd'] = kd
        tag = f'kp{int(kp*100)}_kd{int(kd*100)}'
        trace = os.path.abspath(os.path.join('scripts', f'pid_trace_autotune_{tag}.csv'))
        PID_TRACE['enabled'] = True
        PID_TRACE['path'] = trace
        try:
            if os.path.exists(trace):
                os.remove(trace)
        except Exception:
            pass
        print(f"[AUTO] Running kp={kp:.3f} kd={kd:.3f} -> {trace}")
        sim = simulation(port_name=port, dt=dt, T=T, n_agents=1, load_enc=False, test_mode='zigzag')
        sim.spawn()
        sim.run()
        time.sleep(0.1)
        try:
            df = pd.read_csv(trace)
            df0 = df[df['agent'] == 0]
            total = len(df0)
            max_err = df0['err_deg'].abs().max()
            max_raw = df0['raw_deg'].abs().max()
            mean_rud = df0['rud_deg'].mean()
            max_rud_deg = math.degrees(SHIP_PHYSICS.get('max_rudder', math.radians(20)))
            sat_frac = (df0['rud_deg'].abs() >= (max_rud_deg - 1e-6)).sum() / total if total>0 else 0.0
            results.append({'kp': kp, 'kd': kd, 'max_err': max_err, 'max_raw': max_raw, 'mean_rud': mean_rud, 'sat_frac': sat_frac, 'trace': trace})
            print(f"[AUTO] done: max_err={max_err:.2f}° max_raw={max_raw:.2f}° mean_rud={mean_rud:.2f}° sat_frac={sat_frac:.2%}")
        except Exception as e:
            print('[AUTO] failed to read trace:', e)
            results.append({'kp': kp, 'kd': kd, 'max_err': float('nan'), 'max_raw': float('nan'), 'mean_rud': float('nan'), 'sat_frac': 0.0, 'trace': trace})

# restore
CONTROLLER_GAINS['Kp'] = orig_kp
CONTROLLER_GAINS['Kd'] = orig_kd

print('\nAuto-tune results:')
for r in results:
    print(f"kp={r['kp']:.3f} kd={r['kd']:.3f}  max_err={r['max_err']:6.2f}°  max_raw={r['max_raw']:6.2f}°  mean_rud={r['mean_rud']:6.2f}°  sat_frac={r['sat_frac']:.2%}  trace={r['trace']}")

print('\nDone')
