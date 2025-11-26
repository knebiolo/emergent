"""
Run a small actuator responsiveness sweep: vary SHIP_PHYSICS['max_rudder_rate'] and per-ship
`rudder_tau` and measure closed-loop max heading error. Writes PID trace CSVs and a summary.

Usage: run this script from repository root. It writes files into `scripts/`.
"""
import os
import math
import time
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS, PID_TRACE
from emergent.ship_abm.simulation_core import simulation

# grid (rad/s) and (s)
max_rudder_rates = [0.087, 0.12, 0.17]
rudder_taus = [1.0, 0.5, 0.25]

T = 60.0
dt = 0.1
n_agents = 1
port = 'Rosario Strait'

results = []

orig_max_rate = float(SHIP_PHYSICS.get('max_rudder_rate', 0.087))

for rate in max_rudder_rates:
    for tau in rudder_taus:
        tag = f"rate{str(rate).replace('.','p')}_tau{str(tau).replace('.','p')}"
        trace_path = f'scripts/pid_trace_act_{tag}.csv'
        PID_TRACE['path'] = trace_path
        PID_TRACE['enabled'] = True

        # set global max rate
        SHIP_PHYSICS['max_rudder_rate'] = rate

        print(f"[ACT] Running rate={rate:.3f} rad/s, tau={tau:.3f} s -> {trace_path}")

        # remove old trace
        try:
            if os.path.exists(trace_path):
                os.remove(trace_path)
        except Exception:
            pass

        sim = simulation(port_name=port, dt=dt, T=T, n_agents=n_agents, load_enc=False, test_mode='zigzag')
        sim.spawn()
        # set the ship's actuator time-constant
        try:
            sim.ship.rudder_tau = float(tau)
        except Exception:
            pass

        sim.run()
        time.sleep(0.05)

        # try to read trace and compute metrics
        try:
            df = pd.read_csv(trace_path)
            df0 = df[df['agent'] == 0]
            max_err = df0['err_deg'].abs().max()
            max_raw = df0['raw_deg'].abs().max()
            mean_rud = df0['rud_deg'].mean()
            max_rud = df0['rud_deg'].abs().max()
            results.append({'rate': rate, 'tau': tau, 'max_err_deg': float(max_err), 'max_raw_deg': float(max_raw), 'mean_rud_deg': float(mean_rud), 'max_rud_deg': float(max_rud), 'trace': trace_path})
            print(f"[ACT] max_err={max_err:.2f}°, max_raw={max_raw:.2f}°, mean_rud={mean_rud:.2f}°")
        except Exception as e:
            print(f"[ACT] Failed to read trace {trace_path}: {e}")
            results.append({'rate': rate, 'tau': tau, 'max_err_deg': float('nan'), 'max_raw_deg': float('nan'), 'mean_rud_deg': float('nan'), 'max_rud_deg': float('nan'), 'trace': trace_path})

# restore
SHIP_PHYSICS['max_rudder_rate'] = orig_max_rate

out_df = pd.DataFrame(results)
out_df.to_csv('scripts/actuator_sweep_summary.csv', index=False)
print('\nWrote scripts/actuator_sweep_summary.csv')
