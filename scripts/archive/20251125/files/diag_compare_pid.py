"""Diagnostic: compare baseline vs conservative PID on zig-zag (quiet, short run)

Generates per-run CSVs and a combined PNG of heading, heading command, and rudder.
"""
import importlib.util
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('zig', str(ROOT / 'scripts' / 'zigzag_kf_sweep.py'))
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
run_one = mod.run_one

out_dir = ROOT
cases = [
    {'name':'baseline','wind_speed':0.0,'Kp':0.5,'Ki':0.05,'Kd':0.12,'Kf_gain':0.0,'max_rudder_deg':25.0},
    {'name':'conservative','wind_speed':0.0,'Kp':0.2,'Ki':0.0,'Kd':0.5,'Kf_gain':0.0,'max_rudder_deg':25.0}
]
results = []
for c in cases:
    print('Running', c['name'])
    res = run_one(wind_speed=c['wind_speed'], dead_reck_sens=0.5, Kf_gain=c['Kf_gain'],
                  Kp=c['Kp'], Ki=c['Ki'], Kd=c['Kd'], max_rudder_deg=c['max_rudder_deg'],
                  zig_legs=4, leg_length=150.0, zig_amp=30.0, dt=0.5, T=120.0, verbose=False)
    results.append((c, res))

# read CSVs and plot
fig, axs = plt.subplots(3,1, figsize=(8,9), sharex=True)
for (c,res) in results:
    df = pd.read_csv(res['traj_csv'])
    t = df['t']
    psi = df['psi_deg']
    hd = df['hd_cmd_deg']
    rud = df['rudder_deg']
    label = f"{c['name']} Kp={c['Kp']} Ki={c['Ki']} Kd={c['Kd']}"
    axs[0].plot(t, psi, label=label)
    axs[0].set_ylabel('psi (deg)')
    axs[1].plot(t, hd, label=label)
    axs[1].set_ylabel('hd_cmd (deg)')
    axs[2].plot(t, rud, label=label)
    axs[2].set_ylabel('rudder (deg)')
    axs[2].set_xlabel('t (s)')

for ax in axs:
    ax.legend()
    ax.grid(True)

outp = out_dir / 'figs' / 'zigzag' / 'diag_compare_pid.png'
outp.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(outp)
print('Wrote', outp)
print('Done')
