"""
Quick instrumenter for two specific full-run traces. Writes {stem}_events.csv and {stem}_window.csv under sweep_results/instrumentation
"""
from pathlib import Path
import numpy as np
import pandas as pd

SR = Path('sweep_results')
OUT = SR / 'instrumentation'
OUT.mkdir(parents=True, exist_ok=True)

files = ['pid_trace_osc_keepKp_lowerKd_dtau2.0_full.csv', 'pid_trace_osc_keepKp_lowerKd_dtau2.0_full_nodead.csv']
for fname in files:
    p = SR / fname
    if not p.exists():
        print('missing', p)
        continue
    print('processing', p.name)
    df = pd.read_csv(p)
    # time col
    tcol = 't' if 't' in df.columns else df.columns[0]
    times = pd.to_numeric(df[tcol], errors='coerce').fillna(0.0).values
    # window
    win = df[times <= (times[0] + 60.0)]
    win.to_csv(OUT / (p.stem + '_window.csv'), index=False)
    events = []
    # hd jump
    hd_col = next((c for c in ['hd_cmd_deg','hd_cmd'] if c in df.columns), None)
    if hd_col:
        hd = pd.to_numeric(df[hd_col], errors='coerce').fillna(method='ffill').fillna(0.0).values
        dh = np.abs(np.diff(hd))
        dh = np.minimum(dh, 360.0 - dh)
        idxs = np.where(dh > 90.0)[0]
        for i in idxs:
            events.append({'time': float(times[i]), 'event': 'HD_JUMP', 'info': f'Δdeg={dh[i]:.2f} at idx={i}'})
    # err large
    err_col = next((c for c in ['err_deg','err'] if c in df.columns), None)
    if err_col:
        err = pd.to_numeric(df[err_col], errors='coerce').abs().fillna(0.0).values
        idxs = np.where(err > 30.0)[0]
        for i in idxs:
            events.append({'time': float(times[i]), 'event': 'ERR_LARGE', 'info': f'err_deg={err[i]:.2f} idx={i}'})
    # rud sat
    rud_col = next((c for c in ['rud_deg','rud'] if c in df.columns), None)
    if rud_col:
        rud = pd.to_numeric(df[rud_col], errors='coerce').abs().fillna(0.0).values
        max_obs = np.nanmax(rud)
        threshold = max(0.9 * max_obs, 0.9 * 14.0)
        above = rud >= threshold
        i = 0; n = len(above)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                tdur = float(times[j-1] - times[i])
                if tdur >= 1.0:
                    events.append({'time': float(times[i]), 'event': 'RUD_SAT', 'info': f'dur_s={tdur:.2f} start_idx={i} end_idx={j-1}'})
                i = j
            else:
                i += 1
    ev_df = pd.DataFrame(events)
    ev_df.to_csv(OUT / (p.stem + '_events.csv'), index=False)
    print('wrote', (OUT / (p.stem + '_events.csv')).as_posix(), 'events=', len(events))
print('done')
