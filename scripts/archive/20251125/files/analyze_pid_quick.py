"""Quick PID trace analyzer
Reads scripts/pid_trace_waypoint_crosswind.csv and prints concise diagnostics:
- mean / max / fraction large heading error
- rudder saturation fraction (uses config SHIP_PHYSICS['max_rudder'])
- mean abs raw_deg and rud_deg, r_des stats
"""
import os
import numpy as np
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS

path = os.path.join(os.path.dirname(__file__), 'pid_trace_waypoint_crosswind.csv')
if not os.path.exists(path):
    print('PID trace not found at', path)
    raise SystemExit(1)

df = pd.read_csv(path)

stats = {}
stats['rows'] = len(df)
stats['mean_err_deg'] = df['err_deg'].mean()
stats['mean_abs_err_deg'] = df['err_deg'].abs().mean()
stats['max_abs_err_deg'] = df['err_deg'].abs().max()
stats['frac_err_gt_90'] = float((df['err_deg'].abs() > 90).mean())

# r_des, raw_deg, rud_deg
stats['mean_abs_r_des_deg'] = df['r_des_deg'].abs().mean()
stats['mean_abs_raw_deg'] = df['raw_deg'].abs().mean()
stats['mean_abs_rud_deg'] = df['rud_deg'].abs().mean()
stats['max_abs_rud_deg'] = df['rud_deg'].abs().max()

max_rudder_deg = np.degrees(SHIP_PHYSICS.get('max_rudder', np.radians(20)))
stats['max_rudder_deg_cfg'] = float(max_rudder_deg)
stats['sat_frac'] = float((df['rud_deg'].abs() >= max_rudder_deg * 0.999).mean())

# Print concise summary
print('PID trace:', path)
print(f"rows={stats['rows']}")
print(f"mean_err_deg={stats['mean_err_deg']:.3f}, mean_abs_err_deg={stats['mean_abs_err_deg']:.3f}, max_abs_err_deg={stats['max_abs_err_deg']:.3f}")
print(f"frac_err_gt_90={stats['frac_err_gt_90']:.3%}")
print(f"mean_abs_raw_deg={stats['mean_abs_raw_deg']:.3f}, mean_abs_rud_deg={stats['mean_abs_rud_deg']:.3f}, max_abs_rud_deg={stats['max_abs_rud_deg']:.3f}")
print(f"max_rudder_deg_cfg={stats['max_rudder_deg_cfg']:.3f}, sat_frac={stats['sat_frac']:.3%}")

# Show a few worst rows
nshow = 8
worst = df.loc[df['err_deg'].abs().sort_values(ascending=False).index[:nshow]]
print('\nTop errors (t, err_deg, raw_deg, rud_deg, r_des_deg):')
for _, r in worst.iterrows():
    print(f"t={r['t']:.1f}, err={r['err_deg']:.2f}°, raw={r['raw_deg']:.2f}°, rud={r['rud_deg']:.2f}°, r_des={r['r_des_deg']:.2f}°")
