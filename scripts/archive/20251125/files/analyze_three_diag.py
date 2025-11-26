"""Compare PID trace stats for the three diagnostic scenarios.
"""
import os
import numpy as np
import pandas as pd
from emergent.ship_abm.config import SHIP_PHYSICS

base = os.path.join(os.path.dirname(__file__), 'diag_outputs')
files = [
    'pid_trace_no_wind.csv',
    'pid_trace_light_wind.csv',
    'pid_trace_orig_wind.csv',
]

for fn in files:
    path = os.path.join(base, fn)
    print('\n===', fn, '===')
    if not os.path.exists(path):
        print('missing', path)
        continue
    df = pd.read_csv(path)
    rows = len(df)
    mean_err = df['err_deg'].mean()
    mean_abs_err = df['err_deg'].abs().mean()
    max_abs_err = df['err_deg'].abs().max()
    frac_gt90 = float((df['err_deg'].abs() > 90).mean())
    mean_abs_raw = df['raw_deg'].abs().mean()
    mean_abs_rud = df['rud_deg'].abs().mean()
    max_abs_rud = df['rud_deg'].abs().max()
    max_rudder_deg = np.degrees(SHIP_PHYSICS.get('max_rudder', np.radians(20)))
    sat_frac = float((df['rud_deg'].abs() >= max_rudder_deg * 0.999).mean())

    print('rows=', rows)
    print(f'mean_err_deg={mean_err:.3f}, mean_abs_err_deg={mean_abs_err:.3f}, max_abs_err_deg={max_abs_err:.3f}')
    print(f'frac_err_gt_90={frac_gt90:.3%}')
    print(f'mean_abs_raw_deg={mean_abs_raw:.3f}, mean_abs_rud_deg={mean_abs_rud:.3f}, max_abs_rud_deg={max_abs_rud:.3f}')
    print(f'max_rudder_deg_cfg={max_rudder_deg:.3f}, sat_frac={sat_frac:.3%}')

    # show top error rows
    nshow = 6
    worst = df.loc[df['err_deg'].abs().sort_values(ascending=False).index[:nshow]]
    print('\nTop errors (t, err_deg, raw_deg, rud_deg, r_des_deg):')
    for _, r in worst.iterrows():
        print(f"t={r['t']:.1f}, err={r['err_deg']:.2f}°, raw={r['raw_deg']:.2f}°, rud={r['rud_deg']:.2f}°, r_des={r['r_des_deg']:.2f}°")
