"""Diagnose heading units/column alignment in PID trace CSVs.
Reads two traces:
 - scripts/pid_trace_straight_wind.csv (original)
 - sweep_results/pid_trace_straight_wind_smoothed.csv (smoothed)

For each file:
 - print header, dtypes, min/max
 - infer whether psi column is degrees or radians
 - compute nearest-angle error from hd_cmd and psi
 - compare computed error to logged err_deg (stats + sample diffs)
"""
import math
import pandas as pd

FILES = [
    ('scripts/pid_trace_straight_wind.csv','original'),
    ('sweep_results/pid_trace_straight_wind_smoothed.csv','smoothed')
]


def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0


def infer_psi_units(psi_series):
    # if typical magnitude > 10 -> degrees, else radians
    mx = psi_series.abs().max()
    if mx > 10:
        return 'deg'
    else:
        return 'rad'

for path,label in FILES:
    print('\n====', label, path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print('Failed to read', path, 'error:', e)
        continue

    print('Columns:', list(df.columns))
    print('\nDtypes:\n', df.dtypes)
    df0 = df[df['agent']==0].reset_index(drop=True)
    print('\nRows (n):', len(df0))

    # basic column stats
    for c in ['psi_deg','hd_cmd_deg','err_deg','raw_deg','rud_deg']:
        if c in df0.columns:
            print(f"{c}: min={df0[c].min():.6f} max={df0[c].max():.6f} mean={df0[c].mean():.6f}")

    psi = df0['psi_deg'] if 'psi_deg' in df0.columns else None
    hd = df0['hd_cmd_deg'] if 'hd_cmd_deg' in df0.columns else None
    logged_err = df0['err_deg'] if 'err_deg' in df0.columns else None

    if psi is None or hd is None or logged_err is None:
        print('Missing required columns to diagnose.')
        continue

    units = infer_psi_units(psi)
    print('\nInferred psi units:', units)
    if units == 'rad':
        psi_deg = psi * 180.0 / math.pi
    else:
        psi_deg = psi

    # compute nearest-angle error (hd - psi)
    comp_err = wrap_deg(hd - psi_deg)

    # compare comp_err vs logged err
    diff = comp_err - logged_err
    print('\nComparison comp_err vs logged err (first 10 rows):')
    cmp_df = pd.DataFrame({
        't': df0['t'].iloc[:20],
        'psi_raw': df0['psi_deg'].iloc[:20],
        'psi_deg': psi_deg.iloc[:20],
        'hd_cmd_deg': hd.iloc[:20],
        'logged_err': logged_err.iloc[:20],
        'comp_err': comp_err.iloc[:20],
        'diff': diff.iloc[:20]
    })
    print(cmp_df.to_string(index=False))

    # stats
    print('\nDiff stats: mean_abs_diff=', float(diff.abs().mean()), ' max_abs_diff=', float(diff.abs().max()))

    # sanity checks: if mean_abs_diff is large, try interpreting hd as radians
    if float(diff.abs().mean()) > 1e-3:
        print('\nLarge differences detected: trying alternate interpretations...')
        # try if hd is radians
        alt_hd_deg = None
        if hd.abs().max() < 10:
            alt_hd_deg = hd * 180.0 / math.pi
            alt_comp_err = wrap_deg(alt_hd_deg - psi_deg)
            print('If hd was radians: mean_abs_diff=', float((alt_comp_err - logged_err).abs().mean()))
        # try if both psi and hd are swapped (column misalignment)
        cols = list(df0.columns)
        # attempt shifts of columns by -2..2
        best = None
        for shift in range(-2,3):
            try:
                # pick a candidate psi column by shifting index of 'psi_deg'
                idx = cols.index('psi_deg')
                cand_idx = idx + shift
                if cand_idx < 0 or cand_idx >= len(cols):
                    continue
                cand_col = cols[cand_idx]
                cand_series = df0[cand_col]
                # infer units for candidate
                cand_units = infer_psi_units(cand_series)
                if cand_units == 'rad':
                    cand_deg = cand_series * 180.0 / math.pi
                else:
                    cand_deg = cand_series
                cand_comp_err = wrap_deg(hd - cand_deg)
                err_mean = float((cand_comp_err - logged_err).abs().mean())
                if best is None or err_mean < best[0]:
                    best = (err_mean, shift, cand_col)
            except Exception:
                continue
        if best is not None:
            print('Best alternative column alignment: mean_abs_diff=', best[0], ' shift=', best[1], ' cand_col=', best[2])

print('\nDone')
