import glob
import os
import math
import numpy as np
import pandas as pd

OUT = 'sweep_results'
files = sorted(glob.glob(os.path.join(OUT, 'pid_trace_osc_*.csv')))
rows = []
for f in files:
    df = pd.read_csv(f)
    df0 = df[df['agent']==0]
    if df0.empty:
        continue
    t = df0['t'].values
    err = df0['err_deg'].values
    rud = df0['rud_deg'].values
    mean_err = float(np.mean(np.abs(err)))
    std_err = float(np.std(err))
    pk2pk = float(np.max(err) - np.min(err))
    max_rud = float(np.max(np.abs(rud)))
    # try to infer max_rudder from SHIP_PHYSICS if available, else use max rud in file
    try:
        from emergent.ship_abm.config import SHIP_PHYSICS
        max_rudder_deg = math.degrees(SHIP_PHYSICS['max_rudder'])
    except Exception:
        max_rudder_deg = float('nan')
    sat_frac = float((np.abs(rud) >= max_rudder_deg - 1e-6).sum()) / len(rud) if not math.isnan(max_rudder_deg) else 0.0
    # FFT
    try:
        n = len(err)
        y = err - np.mean(err)
        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(n, d=(t[1]-t[0]) if len(t)>1 else 0.1)
        idx = np.argmax(np.abs(yf[1:])) + 1
        dom_freq = float(xf[idx])
        dom_period = 1.0/dom_freq if dom_freq>0 else float('nan')
        dom_amp = float(np.abs(yf[idx])/n)
    except Exception:
        dom_freq = float('nan')
        dom_period = float('nan')
        dom_amp = float('nan')
    rows.append({
        'file': os.path.basename(f), 'mean_err_deg': mean_err, 'std_err_deg': std_err,
        'pk2pk_err_deg': pk2pk, 'max_rud_deg': max_rud, 'sat_frac': sat_frac,
        'dom_freq_hz': dom_freq, 'dom_period_s': dom_period, 'dom_amp': dom_amp
    })

outf = os.path.join(OUT, 'oscillation_tune_aggregated.csv')
if rows:
    pd.DataFrame(rows).to_csv(outf, index=False)
    print('Wrote', outf)
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(['mean_err_deg', 'std_err_deg']).reset_index(drop=True)
    print(df_sorted.to_string(index=False))
else:
    print('No files found matching', files)
