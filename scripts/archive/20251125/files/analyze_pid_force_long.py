"""
Analyze the long forced PID trace at scripts/pid_trace_force_long.csv and print summary statistics.
"""
import os, sys
import numpy as np
import csv

CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pid_trace_force_long.csv'))
if not os.path.exists(CSV):
    print('CSV not found:', CSV)
    sys.exit(1)

# read header and rows
with open(CSV, 'r', newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader)
    rows = [list(map(float, r)) for r in reader]

arr = np.array(rows)
col_index = {name: i for i, name in enumerate(header)}

err = arr[:, col_index['err_deg']]
r_des = arr[:, col_index['r_des_deg']]
raw = arr[:, col_index['raw_deg']]
rud = arr[:, col_index['rud_deg']]

# compute some metrics
import math

print('Rows:', arr.shape[0])
print('err_deg min/max/mean/std:', (np.min(err), np.max(err), np.mean(err), np.std(err)))
print('raw_deg min/max/mean/std:', (np.min(raw), np.max(raw), np.mean(raw), np.std(raw)))
print('rud_deg min/max/mean/std:', (np.min(rud), np.max(rud), np.mean(rud), np.std(rud)))

# peak yaw rate implied by applied rudder approximated via linear scaling: small-approx
# not exact; for now compute fraction of rows where |rud| > 5°, 10°, 15°
for thresh in [5.0, 10.0, 15.0]:
    frac = np.mean(np.abs(rud) >= thresh)
    print(f'fraction of steps with |rud| >= {thresh}°: {frac:.3f}')

# show a few sample rows (first 10)
print('\nSample rows:')
for r in rows[:10]:
    print(dict(zip(header, r)))
