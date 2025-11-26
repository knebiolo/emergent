"""
Post-process sweep trace CSVs to locate failure events.

- Scans `sweep_results/` for files named `*_trace*.csv` or `*_pid_trace*.csv`.
- For each trace, writes:
  - `sweep_results/instrumentation/{runname}_events.csv` containing detected events.
  - `sweep_results/instrumentation/{runname}_window.csv` first 60s of the trace for quick inspection.

Event detection heuristics:
- HD_JUMP: commanded heading (`hd_cmd_deg` or `hd_cmd`) jumps by >90° between adjacent samples.
- ERR_LARGE: |err_deg| > 30° at any sample.
- RUD_SAT: rudder (`rud_deg` or `rud`) has absolute value >= (max_rudder_deg - 0.5°) for >1s continuous.

Run as a standalone script from project root. Outputs are CSVs and prints a short summary.
"""
from pathlib import Path
import csv
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / 'sweep_results'
OUT = SR / 'instrumentation'
OUT.mkdir(parents=True, exist_ok=True)

# candidate filenames
trace_files = list(SR.glob('*trace*.csv')) + list(SR.glob('*_trace*.csv'))
trace_files = [p for p in trace_files if p.is_file()]

if not trace_files:
    print('No trace CSVs found in sweep_results/.')
    raise SystemExit(0)

summary_rows = []

for tf in trace_files:
    runname = tf.stem
    try:
        df = pd.read_csv(tf)
    except Exception as e:
        print(f'Failed to read {tf}: {e}')
        continue

    # try to find time column
    tcol = None
    for c in ['time', 't', 'timestamp', 'secs']:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # assume first column is time-like
        tcol = df.columns[0]

    # normalize column names mapping
    def find_col(possible):
        for p in possible:
            if p in df.columns:
                return p
        return None

    hd_col = find_col(['hd_cmd_deg', 'hd_cmd', 'hd_cmd_deg.1'])
    psi_col = find_col(['psi_deg', 'psi', 'heading_deg'])
    err_col = find_col(['err_deg', 'err', 'err_deg.1'])
    rud_col = find_col(['rud_deg', 'rud', 'rudder_deg', 'rudder'])

    # coerce time to numeric seconds (if it's a formatted string, try to parse float)
    try:
        times = pd.to_numeric(df[tcol], errors='coerce').values
        # replace NaN with incremental indices if parsing failed entirely
        if np.all(np.isnan(times)):
            times = np.arange(len(df)) * 0.1
            times = times.astype(float)
        # fill any leading NaNs with nearest valid or 0
        nan_mask = np.isnan(times)
        if nan_mask.any():
            times[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), times[~nan_mask])
    except Exception:
        times = np.arange(len(df)) * 0.1
        times = times.astype(float)
    # limit to first 60s window
    try:
        window_mask = times <= (times[0] + 60.0)
    except Exception:
        window_mask = np.arange(len(df)) < min(len(df), 600)
    df_window = df[window_mask]
    win_out = OUT / f"{runname}_window.csv"
    try:
        df_window.to_csv(win_out, index=False)
    except Exception:
        pass

    events = []

    # HD_JUMP detection from hd_col
    if hd_col is not None:
        hd = pd.to_numeric(df[hd_col], errors='coerce').fillna(method='ffill').fillna(0.0).values
        # ensure in degrees
        # detect big jumps between adjacent samples
        dh = np.abs(np.diff(hd))
        # account for wrap-around >180
        dh = np.minimum(dh, 360.0 - dh)
        idxs = np.where(dh > 90.0)[0]
        for i in idxs:
            t0 = float(times[i])
            events.append({'time': t0, 'event': 'HD_JUMP', 'info': f'Δdeg={dh[i]:.2f} at idx={i}'})

    # ERR_LARGE detection
    if err_col is not None:
        err = pd.to_numeric(df[err_col], errors='coerce').abs().fillna(0.0).values
        idxs = np.where(err > 30.0)[0]
        for i in idxs:
            events.append({'time': float(times[i]), 'event': 'ERR_LARGE', 'info': f'err_deg={err[i]:.2f} idx={i}'})

    # RUD_SAT detection: need to know max_rudder_deg; attempt to read header comment or config file
    # fallback threshold: 90% of max observed rudder in trace or 90% of 14deg
    if rud_col is not None:
        rud = pd.to_numeric(df[rud_col], errors='coerce').abs().fillna(0.0).values
        # find candidate threshold
        max_obs = np.nanmax(rud)
        threshold = max(0.9 * max_obs, 0.9 * 14.0)
        # find contiguous segments above threshold
        above = rud >= threshold
        if above.any():
            # find runs
            i = 0
            n = len(above)
            while i < n:
                if above[i]:
                    j = i
                    while j < n and above[j]:
                        j += 1
                    # duration
                    tdur = float(times[j-1] - times[i])
                    if tdur >= 1.0:
                        events.append({'time': float(times[i]), 'event': 'RUD_SAT', 'info': f'dur_s={tdur:.2f} start_idx={i} end_idx={j-1}'})
                    i = j
                else:
                    i += 1

    # write events CSV
    ev_df = pd.DataFrame(events)
    ev_out = OUT / f"{runname}_events.csv"
    try:
        ev_df.to_csv(ev_out, index=False)
    except Exception:
        pass

    summary_rows.append({'run': runname, 'n_events': len(events)})

# write global summary
sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv(OUT / 'instrumentation_summary.csv', index=False)
print('Instrumentation pass complete. Results in', OUT)
