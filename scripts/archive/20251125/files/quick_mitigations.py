"""Quick mitigation experiments for zig-zag oscillation diagnosis.

Runs a set of short zig-zag cases with conservative controller modifications:
 - reduce Kp
 - set Ki=0
 - increase Kd
 - increase backcalc_beta
 - increase max rudder rate

Produces a summary CSV and a combined PNG under `figs/zigzag`.
"""
from pathlib import Path
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('zig', str(ROOT / 'scripts' / 'zigzag_kf_sweep.py'))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
run_one = mod.run_one

out_dir = ROOT / 'figs' / 'zigzag'
out_dir.mkdir(parents=True, exist_ok=True)

cases = [
    {'name':'baseline','Kp':0.5,'Ki':0.05,'Kd':0.12,'Kf_gain':0.0,'max_rudder_deg':25.0,'backcalc_beta':0.16,'deriv_tau':1.0,'max_rudder_rate':0.087},
    {'name':'lowP_noI','Kp':0.25,'Ki':0.0,'Kd':0.2,'Kf_gain':0.0,'max_rudder_deg':25.0,'backcalc_beta':0.3,'deriv_tau':1.5,'max_rudder_rate':0.087},
    {'name':'highD','Kp':0.4,'Ki':0.0,'Kd':0.6,'Kf_gain':0.0,'max_rudder_deg':25.0,'backcalc_beta':0.2,'deriv_tau':0.5,'max_rudder_rate':0.087},
    {'name':'fast_actuator','Kp':0.4,'Ki':0.0,'Kd':0.4,'Kf_gain':0.0,'max_rudder_deg':25.0,'backcalc_beta':0.2,'deriv_tau':0.7,'max_rudder_rate':0.35},
]

results = []
for c in cases:
    print('Running', c['name'])
    # run short zig-zag
    res = run_one(wind_speed=0.0, dead_reck_sens=0.5, Kf_gain=c['Kf_gain'],
                  Kp=c['Kp'], Ki=c['Ki'], Kd=c['Kd'], max_rudder_deg=c['max_rudder_deg'],
                  zig_legs=4, leg_length=150.0, zig_amp=30.0,
                  dt=0.5, T=120.0, verbose=False,
                  deriv_tau=c.get('deriv_tau', None), backcalc_beta=c.get('backcalc_beta', None), max_rudder_rate=c.get('max_rudder_rate', None))
    # add tags for plotting
    res['case'] = c['name']
    res['deriv_tau'] = c['deriv_tau']
    res['backcalc_beta'] = c['backcalc_beta']
    res['max_rudder_rate'] = c['max_rudder_rate']
    results.append(res)

# write summary CSV
summary_csv = Path.cwd() / 'quick_mitigations_summary.csv'
pd.DataFrame(results).to_csv(summary_csv, index=False)
print('Wrote', summary_csv)

# plot comparison: psi, hd_cmd, rudder for each case
fig, axs = plt.subplots(3,1, figsize=(8,10), sharex=True)
for r in results:
    df = pd.read_csv(r['traj_csv'])
    t = df['t']
    psi = df['psi_deg']
    hd = df['hd_cmd_deg']
    rud = df['rudder_deg']
    label = f"{r['case']} Kp={r['Kp']} Ki={r['Ki']} Kd={r['Kd']}"
    axs[0].plot(t, psi, label=label)
    axs[1].plot(t, hd, label=label)
    axs[2].plot(t, rud, label=label)

for ax in axs:
    ax.grid(True)
    ax.legend()

axs[0].set_ylabel('psi (deg)')
axs[1].set_ylabel('hd_cmd (deg)')
axs[2].set_ylabel('rudder (deg)')
axs[2].set_xlabel('t (s)')

outp = out_dir / 'quick_mitigations_compare.png'
fig.tight_layout()
fig.savefig(outp)
print('Wrote', outp)

print('Done')
