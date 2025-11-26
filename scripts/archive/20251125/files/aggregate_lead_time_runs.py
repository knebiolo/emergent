"""Aggregate existing lead_time pid_trace_lead20_* files into sweep_results/lead_time_max_rudder_summary.csv
"""
import glob
import os
import math
import numpy as np
import pandas as pd

OUT = 'sweep_results'
files = sorted(glob.glob(os.path.join(OUT, 'pid_trace_lead20_mr*_rel*.csv')))
rows = []
for f in files:
    df = pd.read_csv(f)
    df0 = df[df['agent']==0]
    t = df0['t'].values
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values
    mean_err = float(np.mean(np.abs(err)))
    std_err = float(np.std(err))
    pk2pk = float(np.max(err) - np.min(err))
    max_rud = float(np.max(np.abs(rud)))
    # infer max_rudder_deg from filename
    import re
    m = re.search(r'mr(\d+)_', os.path.basename(f))
    mr = float(m.group(1)) if m else float('nan')
    sat_frac = float((np.abs(rud) >= (mr - 1e-6)).sum()) / len(rud) if not math.isnan(mr) else 0.0
    rows.append({'name': os.path.basename(f).replace('pid_trace_','').replace('.csv',''),
                 'max_rudder_deg': mr, 'mean_err_deg': mean_err, 'std_err_deg': std_err,
                 'pk2pk_err_deg': pk2pk, 'max_rud_deg': max_rud, 'sat_frac': sat_frac, 'trace': f})

outf = os.path.join(OUT, 'lead_time_max_rudder_summary.csv')
if rows:
    pd.DataFrame(rows).to_csv(outf, index=False)
    print('Wrote', outf)
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print('No files found')
