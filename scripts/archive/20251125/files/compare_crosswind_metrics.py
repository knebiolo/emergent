"""Compare nearest-angle metrics between original and smoothed straight-line crosswind traces.
Reads:
  scripts/pid_trace_straight_wind.csv (original)
  sweep_results/pid_trace_straight_wind_smoothed.csv (smoothed)
Prints a small table to stdout.
"""
import os
import pandas as pd

def wrap_deg_to_signed(a_deg):
    return ((a_deg + 180.0) % 360.0) - 180.0

paths = {
    'original': 'scripts/pid_trace_straight_wind.csv',
    'smoothed': 'sweep_results/pid_trace_straight_wind_smoothed.csv'
}

results = {}
for k,p in paths.items():
    if not os.path.exists(p):
        results[k] = None
        continue
    df = pd.read_csv(p)
    df0 = df[df['agent']==0].reset_index(drop=True)
    hd = df0['hd_cmd_deg']
    psi = df0['psi_deg']
    rud = df0['rud_deg']
    hd_wr = wrap_deg_to_signed(hd)
    psi_wr = wrap_deg_to_signed(psi)
    err = wrap_deg_to_signed(hd_wr - psi_wr).abs()
    results[k] = {
        'n': len(err),
        'max_err': float(err.max()),
        'mean_err': float(err.mean()),
        'frac_err_gt_30': float((err>30.0).sum())/len(err),
        'max_rud': float(rud.abs().max()),
        'frac_rud_at_max': float((rud.abs()>=(rud.abs().max()-1e-6)).sum())/len(rud)
    }

print('Comparison (nearest-angle errors):')
print('file                n    max_err   mean_err  frac_err>30  max_rud  frac_rud_at_max')
for k in ['original','smoothed']:
    v = results[k]
    if v is None:
        print(f'{k:16s} MISSING')
    else:
        print(f'{k:16s} {v["n"]:5d} {v["max_err"]:8.2f} {v["mean_err"]:9.2f} {v["frac_err_gt_30"]:11.3f} {v["max_rud"]:8.2f} {v["frac_rud_at_max"]:14.3f}')
