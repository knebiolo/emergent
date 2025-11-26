"""
Focused sweep over Kd and deriv_tau for long zigzag tests.
Creates per-run PID trace CSVs in `traces/` and a summary CSV `zigzag_focus_summary.csv`.
"""
import os
import itertools
import numpy as np
from emergent.ship_abm import config
from emergent.ship_abm.config import PID_TRACE
from emergent.ship_abm.simulation_core import simulation, compute_zigzag_metrics

# Sweep parameters
Kd_vals = [1.0, 2.0, 3.0]
deriv_tau_vals = [0.5, 1.0, 2.0]

T = 360.0
dt = 0.5
zigdeg = 10
hold = 60

OUT_DIR = 'traces'
os.makedirs(OUT_DIR, exist_ok=True)

summary_rows = []

for kd, tau in itertools.product(Kd_vals, deriv_tau_vals):
    tag = f"kd{kd:.1f}_tau{tau:.1f}".replace('.', 'p')
    trace_path = os.path.join(OUT_DIR, f"pid_trace_{tag}.csv")

    # Apply tuning
    config.SHIP_AERO_DEFAULTS['wind_force_scale'] = 0.35
    config.CONTROLLER_GAINS['Kd'] = kd
    config.ADVANCED_CONTROLLER['deriv_tau'] = tau
    config.ADVANCED_CONTROLLER['trim_band_deg'] = 2.0
    config.ADVANCED_CONTROLLER['lead_time'] = 5.0

    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = trace_path

    print(f"Running long zigzag for {tag} -> {trace_path}")
    sim = simulation(port_name='Baltimore', dt=dt, T=T, n_agents=1, load_enc=False,
                     test_mode='zigzag', zigzag_deg=zigdeg, zigzag_hold=hold)
    sim.spawn()
    sim.run()

    t = np.array(sim.t_history)
    psi = np.array(sim.psi_history)
    hd = np.array(sim.hd_cmd_history)
    metrics = compute_zigzag_metrics(t, psi, hd, tol=3.0)

    row = {
        'tag': tag,
        'Kd': kd,
        'deriv_tau': tau,
        'peak_overshoot_deg': metrics.get('peak_overshoot_deg', np.nan),
        'settling_time_s': metrics.get('settling_time_s', np.nan),
        'steady_state_error_deg': metrics.get('steady_state_error_deg', np.nan),
        'oscillation_period_s': metrics.get('oscillation_period_s', np.nan),
        'trace_path': trace_path
    }
    summary_rows.append(row)

# Write summary CSV
import csv
summary_path = 'zigzag_focus_summary.csv'
with open(summary_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    for r in summary_rows:
        writer.writerow(r)

print('Sweep complete. Summary ->', summary_path)
