"""Compare PID diagnostics across multiple PID trace CSVs.
Usage: python scripts/compare_pid_diagnostics.py file1.csv file2.csv ...
If no args provided, it will compare a sensible default set.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DEFAULTS = [
    'scripts/pid_trace_repro_t478.5.csv',
    'scripts/pid_trace_forced.csv',
    'scripts/pid_trace_tuned.csv'
]

files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULTS

results = []
for f in files:
    p = Path(f)
    if not p.exists():
        print(f"Skipping missing file: {f}")
        continue
    df = pd.read_csv(p)
    if 't' not in df.columns:
        print(f"Skipping {f}: no 't' column")
        continue
    times = df['t'].to_numpy()
    dt = np.median(np.diff(times)) if len(times) > 1 else 0.5
    err = df['err_deg'].to_numpy() if 'err_deg' in df.columns else np.zeros_like(times)
    I = df['I_deg'].to_numpy() if 'I_deg' in df.columns else np.zeros_like(times)
    raw = df['raw_deg'].to_numpy() if 'raw_deg' in df.columns else np.zeros_like(times)
    rud = df['rud_deg'].to_numpy() if 'rud_deg' in df.columns else np.zeros_like(times)

    peak_err = float(np.max(np.abs(err)))
    max_I = float(np.max(np.abs(I)))
    max_raw = float(np.max(np.abs(raw)))
    max_rud = float(np.max(np.abs(rud)))
    sat_mask = np.isclose(np.abs(rud), max_rud, atol=1e-6) | (np.abs(rud) > (max_rud - 1e-6))
    sat_time_s = float(sat_mask.sum() * dt)
    first_sat_t = float(times[sat_mask.argmax()]) if sat_mask.any() else None

    results.append((f, peak_err, max_I, max_raw, max_rud, sat_time_s, first_sat_t))

# print table
if not results:
    print('No files processed')
    sys.exit(0)

print('\nDiagnostics comparison:')
print('file, peak_err_deg, max_I_deg, max_raw_deg, max_rud_deg, sat_time_s, first_sat_t')
for r in results:
    print(','.join([str(x) for x in r]))

# simple interpretation: for each file compare to first
base = results[0]
print('\nRelative to', base[0])
for r in results[1:]:
    print('\nFile:', r[0])
    print(f"  peak_err: {r[1]:.3f} (delta {r[1]-base[1]:+.3f})")
    print(f"  max_I:    {r[2]:.3f} (delta {r[2]-base[2]:+.3f})")
    print(f"  max_raw:  {r[3]:.3f} (delta {r[3]-base[3]:+.3f})")
    print(f"  max_rud:  {r[4]:.3f} (delta {r[4]-base[4]:+.3f})")
    print(f"  sat_time: {r[5]:.3f} (delta {r[5]-base[5]:+.3f})")
    print(f"  first_sat_t: {r[6]} (base {base[6]})")
