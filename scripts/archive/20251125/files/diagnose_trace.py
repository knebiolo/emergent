"""Diagnose a PID trace CSV and print concise time-series diagnostics.

Usage: python scripts/diagnose_trace.py <trace_path>

Writes a small summary CSV under sweep_results/diagnostics_<basename>.csv
and prints key lines to stdout.
"""
import sys
import os
import math
import pandas as pd

if len(sys.argv) < 2:
    print('Usage: python scripts/diagnose_trace.py <trace_path>')
    sys.exit(1)

trace = sys.argv[1]
OUTDIR = 'sweep_results'
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(trace)
# pick agent 0 if present
if 'agent' in df.columns:
    df = df[df['agent'] == 0]

N = len(df)
if N == 0:
    print('Empty trace after filtering agent==0')
    sys.exit(1)

# basics
t0 = float(df['t'].iloc[0])
tN = float(df['t'].iloc[-1])
dur = tN - t0
err = df['err_deg'].astype(float).values
abs_err = np_abs_err = (abs(err))
mean_abs = float(abs_err.mean())
max_err = float(abs_err.max())
std_err = float(abs_err.std())
pk2pk = float(err.max() - err.min())

# thresholds
th1 = 30.0
th2 = 90.0
frac_over_th1 = float((abs_err > th1).sum()) / N
frac_over_th2 = float((abs_err > th2).sum()) / N
first_over_th1 = float(df['t'].values[(abs_err > th1).argmax()]) if (abs_err > th1).any() else None

# rudder
if 'rud_deg' in df.columns:
    rud = df['rud_deg'].astype(float).values
else:
    rud = None
max_rud = float(abs(rud).max()) if rud is not None else None
# use sim max_rudder if present in trace header? fall back to NaN
# fraction of time with applied rudder at/near typical max_rudder (14 deg) threshold
max_rudder_config = 14.0
sat_frac = float((abs(rud) >= (max_rudder_config - 1e-6)).sum()) / N if rud is not None else None

# sample early timeframe for investigation
sample = df.head(50)[['t','err_deg','r_des_deg','raw_deg','rud_deg','psi_deg','hd_cmd_deg']]

summary = {
    'trace': trace,
    'rows': N,
    'duration_s': dur,
    'mean_abs_err_deg': mean_abs,
    'max_abs_err_deg': max_err,
    'std_err_deg': std_err,
    'pk2pk_err_deg': pk2pk,
    'frac_err_abs_gt_30_deg': frac_over_th1,
    'frac_err_abs_gt_90_deg': frac_over_th2,
    'first_time_err_gt_30s': first_over_th1,
    'max_rud_deg': max_rud,
    'sat_frac_by_14deg': sat_frac
}

out_csv = os.path.join(OUTDIR, 'diagnostics_' + os.path.basename(trace).replace('.csv','.csv'))
# write sample + header summary as CSV (sample rows then a blank line then summary key-values)
sample.to_csv(out_csv, index=False)
with open(out_csv,'a') as f:
    f.write('\n')
    for k,v in summary.items():
        f.write(f'{k},{v}\n')

# print concise diagnostics
print('Diagnostics for', trace)
for k in ['rows','duration_s','mean_abs_err_deg','max_abs_err_deg','std_err_deg','pk2pk_err_deg','frac_err_abs_gt_30_deg','frac_err_abs_gt_90_deg','first_time_err_gt_30s','max_rud_deg','sat_frac_by_14deg']:
    print(f'  {k}: {summary[k]}')

print('\nEarly sample (first 10 rows):')
print(sample.head(10).to_string(index=False))
print('\nWrote', out_csv)
