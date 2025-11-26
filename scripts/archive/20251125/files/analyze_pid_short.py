"""
Analyze the short PID trace at scripts/pid_trace_short.csv and print summary statistics.
"""
import os, sys
import numpy as np
import csv

CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pid_trace_short.csv'))
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

# important columns (best-effort matching)
for name in ['t','agent','err_deg','r_des_deg','derr_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']:
    if name not in col_index:
        print('Missing expected column in CSV:', name)

err = arr[:, col_index['err_deg']]
r_des = arr[:, col_index.get('r_des_deg', col_index['derr_deg'])]  # fallback
raw = arr[:, col_index['raw_deg']]
rud = arr[:, col_index['rud_deg']]

def stats(x):
    return np.min(x), np.max(x), np.mean(x), np.std(x)

print('Rows:', arr.shape[0])
print('err_deg min/max/mean/std:', stats(err))
print('r_des_deg min/max/mean/std:', stats(r_des))
print('raw_deg min/max/mean/std:', stats(raw))
print('rud_deg min/max/mean/std:', stats(rud))

# saturation detection: where raw > max rudder (approx 20 deg) in magnitude
max_rud = 20.0
sat = np.abs(raw) >= max_rud - 1e-6
print('Saturation count:', np.sum(sat), 'fraction:', np.mean(sat))

# fraction of time final applied rud is at limit
sat_applied = np.abs(rud) >= max_rud - 1e-6
print('Applied saturation count:', np.sum(sat_applied), 'fraction:', np.mean(sat_applied))

# show first 5 rows
print('\nFirst 5 rows:')
for r in rows[:5]:
    print(dict(zip(header, r)))
