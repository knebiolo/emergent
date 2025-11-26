"""Analyze instrumented straight-line runs.
Computes Pearson correlation, cross-correlation lag (s), RMS/MAE of command-applied, saturation stats,
and saves overlay plots plus a summary CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent.parent
IN = ROOT / 'figs' / 'straight_line_instrumented'
OUT = IN
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(IN.glob('straight_instr_*.csv'))
rows = []

def estimate_lag_seconds(cmd, applied, dt):
    # remove nan and zero-mean
    mask = ~np.isnan(cmd) & ~np.isnan(applied)
    if mask.sum() < 3:
        return np.nan
    c = cmd[mask] - np.mean(cmd[mask])
    a = applied[mask] - np.mean(applied[mask])
    corr = correlate(a, c, mode='full')
    lag_idx = np.argmax(corr) - (c.size - 1)
    return lag_idx * dt

for f in files:
    df = pd.read_csv(f)
    t = df['t_s'].values
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.5
    cmd = df['rud_cmd_deg'].values if 'rud_cmd_deg' in df.columns else np.full_like(t, np.nan)
    applied = df['rud_applied_deg'].values if 'rud_applied_deg' in df.columns else np.full_like(t, np.nan)

    # align
    mask = ~np.isnan(cmd) & ~np.isnan(applied)
    if mask.sum() < 3:
        corr = np.nan; lag_s = np.nan; rms_err = np.nan; mae = np.nan
    else:
        try:
            corr, _ = pearsonr(cmd[mask], applied[mask])
        except Exception:
            corr = np.nan
        lag_s = estimate_lag_seconds(cmd, applied, dt)
        err = cmd[mask] - applied[mask]
        rms_err = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))

    # saturation proxies
    appl_abs = np.abs(applied[~np.isnan(applied)])
    if appl_abs.size == 0:
        pct_high = np.nan; appl_max = np.nan
    else:
        appl_max = float(np.nanmax(appl_abs))
        # fraction of samples within 98% of observed max (proxy for saturation)
        pct_high = float((appl_abs >= 0.98 * appl_max).sum() / appl_abs.size)

    # save overlay plot
    tag = f.stem.replace('straight_instr_', '')
    plt.figure(figsize=(10,4))
    plt.plot(t, cmd, label='cmd_rud (deg)')
    plt.plot(t, applied, label='applied_rud (deg)')
    plt.title(f'Rudder: {tag} | corr={corr:.3f} lag_s={lag_s!s}')
    plt.legend(); plt.grid(True)
    plt.xlabel('t (s)')
    png = OUT / f'analysis_rudder_{tag}.png'
    plt.savefig(png, dpi=150)
    plt.close()

    # error plot
    plt.figure(figsize=(10,3))
    plt.plot(t, cmd - applied, label='cmd - applied (deg)')
    plt.axhline(0, color='k', lw=0.5)
    plt.title(f'Cmd-Applied Error: {tag} | RMS={rms_err:.3f} deg')
    plt.grid(True)
    plt.xlabel('t (s)')
    png2 = OUT / f'analysis_rudder_error_{tag}.png'
    plt.savefig(png2, dpi=150)
    plt.close()

    rows.append({
        'file': str(f), 'tag': tag, 'dt_s': dt,
        'pearson_cmd_applied': corr, 'lag_s': lag_s,
        'rms_cmd_minus_applied_deg': rms_err, 'mae_deg': mae,
        'applied_max_deg': appl_max, 'pct_applied_near_max': pct_high,
        'overlay_png': str(png), 'error_png': str(png2)
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT / 'analysis_straight_line_instrumented_summary.csv', index=False)
print('Wrote analysis summary to', OUT / 'analysis_straight_line_instrumented_summary.csv')
