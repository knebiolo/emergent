import csv
from math import fabs
p = r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\scripts\pid_trace_gui_rosario.csv"
max_err = (0.0, None)
max_raw = (0.0, None)
max_rud = (0.0, None)
big = []
with open(p, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            t = float(row['t'])
            err = fabs(float(row['err_deg']))
            raw = fabs(float(row['raw_deg']))
            rud = fabs(float(row['rud_deg']))
        except Exception:
            continue
        if err > max_err[0]: max_err = (err, t)
        if raw > max_raw[0]: max_raw = (raw, t)
        if rud > max_rud[0]: max_rud = (rud, t)
        if raw > 5.0 or rud > 5.0:
            big.append((t, row['err_deg'], row['raw_deg'], row['rud_deg']))
print(f"max_err_deg={max_err[0]:.3f} at t={max_err[1]}")
print(f"max_raw_deg={max_raw[0]:.3f} at t={max_raw[1]}")
print(f"max_rud_deg={max_rud[0]:.3f} at t={max_rud[1]}")
print(f"count rows with |raw|>5 or |rud|>5: {len(big)}")
if big:
    print('\nFirst 20 large rows:')
    for row in big[:20]:
        print(row)
