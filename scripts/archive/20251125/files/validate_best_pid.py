import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from emergent.ship_abm.config import CONTROLLER_GAINS, PID_TRACE
from emergent.ship_abm.simulation_core import simulation

best = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.3}
CONTROLLER_GAINS.update(best)

trace_path = 'scripts/pid_trace_best_validation.csv'
PID_TRACE['path'] = trace_path
PID_TRACE['enabled'] = True

sim = simulation(port_name='Rosario Strait', dt=0.1, T=120.0, n_agents=1, load_enc=False, test_mode='zigzag')
sim.spawn()
sim.run()
time.sleep(0.05)

if os.path.exists(trace_path):
    df = pd.read_csv(trace_path)
    df0 = df[df['agent'] == 0]
    t = df0['t'].values
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values
    raw = df0['raw_deg'].values

    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(t, err, label='heading error (deg)')
    plt.xlabel('t (s)')
    plt.ylabel('deg')
    plt.title('Validation: heading error (best PID)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    png1 = 'plots/final_heading.png'
    plt.savefig(png1)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(t, rud, label='applied rudder (deg)')
    plt.plot(t, raw, '--', label='raw pid cmd (deg)', alpha=0.7)
    plt.xlabel('t (s)')
    plt.ylabel('deg')
    plt.title('Validation: rudder (best PID)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    png2 = 'plots/final_rudder.png'
    plt.savefig(png2)
    plt.close()

    # print numeric summary
    print('Validation summary:')
    print(' max_err_deg:', float(df0['err_deg'].abs().max()))
    print(' mean_err_deg:', float(df0['err_deg'].abs().mean()))
    print(' mean_rud_deg:', float(df0['rud_deg'].mean()))
    print('\nSaved plots:', png1, png2)
else:
    print('Trace not found:', trace_path)
