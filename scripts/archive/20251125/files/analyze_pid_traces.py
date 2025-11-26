import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from emergent.ship_abm import config
except Exception as e:
    # fallback: try relative import
    try:
        import emergent.ship_abm.config as config
    except Exception:
        config = None

DATA_GLOB = Path(__file__).resolve().parents[0] / 'pid_trace_expt_*.csv'
files = sorted(Path(__file__).resolve().parents[0].glob('pid_trace_expt_*.csv'))
if not files:
    print('No pid_trace_expt_*.csv files found in scripts/ — run the experiment first')
    sys.exit(2)

rows = []
for f in files:
    df = pd.read_csv(f)
    # basic checks
    if 't' not in df.columns:
        print(f'File {f} missing t column, skipping')
        continue
    dt = np.median(np.diff(df['t'].values)) if len(df) > 1 else np.nan

    max_err = df['err_deg'].abs().max()
    max_raw = df['raw_deg'].abs().max()
    max_rud = df['rud_deg'].abs().max()
    max_diff = (df['raw_deg'] - df['rud_deg']).abs().max()
    mean_diff = (df['raw_deg'] - df['rud_deg']).abs().mean()

    # get max_rudder from config (degrees)
    max_rudder_deg = None
    if config and hasattr(config, 'SHIP_PHYSICS'):
        try:
            max_rudder_deg = np.degrees(config.SHIP_PHYSICS.get('max_rudder'))
        except Exception:
            max_rudder_deg = None

    # saturation fraction: rows where applied rudder magnitude >= max_rudder_deg*0.999
    sat_frac = None
    if max_rudder_deg is not None and not np.isnan(max_rudder_deg):
        sat_frac = np.mean(df['rud_deg'].abs() >= (max_rudder_deg * 0.999))
    else:
        sat_frac = np.nan

    # raw exceed fraction (raw command larger than hardware limit)
    raw_exceed_frac = None
    if max_rudder_deg is not None and not np.isnan(max_rudder_deg):
        raw_exceed_frac = np.mean(df['raw_deg'].abs() > (max_rudder_deg + 1e-6))

    # approx mean applied rudder rate (deg/s)
    rud_rate = np.nan
    if len(df) > 2 and not np.isnan(dt) and dt > 0:
        rud_rate = np.mean(np.abs(np.diff(df['rud_deg'].values) / dt))

    # times of largest errors
    idx = df['err_deg'].abs().idxmax()
    t_max_err = df.loc[idx, 't']
    raw_at_max = df.loc[idx, 'raw_deg']
    rud_at_max = df.loc[idx, 'rud_deg']

    rows.append({
        'file': str(f.name),
        'n_rows': len(df),
        'dt_est_s': float(dt),
        'max_err_deg': float(max_err),
        'max_raw_deg': float(max_raw),
        'max_rud_deg': float(max_rud),
        'max_abs_diff_deg': float(max_diff),
        'mean_abs_diff_deg': float(mean_diff),
        'approx_mean_rud_deg_per_s': float(rud_rate) if not np.isnan(rud_rate) else None,
        'sat_frac': float(sat_frac) if not np.isnan(sat_frac) else None,
        'raw_exceed_frac': float(raw_exceed_frac) if raw_exceed_frac is not None else None,
        't_max_err_s': float(t_max_err),
        'raw_at_max_err_deg': float(raw_at_max),
        'rud_at_max_err_deg': float(rud_at_max),
        'max_rudder_limit_deg': float(max_rudder_deg) if max_rudder_deg is not None else None,
    })

out_df = pd.DataFrame(rows)
print(out_df.to_string(index=False))

out_csv = Path(__file__).resolve().parents[0] / 'pid_trace_expt_summary.csv'
out_df.to_csv(out_csv, index=False)
print('\nWrote summary to', out_csv)
