import csv
import math
from statistics import mean, stdev

path = 'scripts/pid_trace_forced.csv'
cols = ['err_deg','P_deg','I_deg','D_deg','raw_deg','rud_deg']
vals = {c: [] for c in cols}
count = 0
with open(path, newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        count += 1
        for c in cols:
            try:
                vals[c].append(float(row[c]))
            except Exception:
                vals[c].append(0.0)

print('PID CSV:', path)
print('rows:', count)
for c in cols:
    arr = vals[c]
    if not arr:
        print(f"{c}: no data")
        continue
    mn = mean(arr)
    try:
        sd = stdev(arr)
    except Exception:
        sd = 0.0
    mnv = min(arr)
    mxv = max(arr)
    print(f"{c}: mean={mn:.4f}, std={sd:.4f}, min={mnv:.4f}, max={mxv:.4f}")
