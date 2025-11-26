import os
import glob
import csv
import numpy as np

def parse_params_from_name(fn):
    # expected pattern: headless_dead_reckoning_traj_w{w}_dr{dr}_Kf{Kf}.csv
    b = os.path.basename(fn)
    parts = b.replace('.csv','').split('_')
    out = {}
    for p in parts:
        if p.startswith('w') and p[1:].replace('.','',1).isdigit():
            out['wind'] = float(p[1:])
        if p.startswith('dr'):
            out['dr'] = float(p[2:])
        if p.startswith('Kf'):
            out['Kf'] = float(p[2:])
    return out

def summarize():
    files = glob.glob(os.path.join(os.getcwd(),'headless_dead_reckoning_traj_*.csv'))
    rows = []
    for f in files:
        params = parse_params_from_name(f)
        # read cross_track_m column (last column)
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        if data.ndim == 1:
            ct = np.array([data[-1]])
        else:
            ct = data[:,-1]
        final = float(ct[-1])
        rmse = float(np.sqrt(np.mean(ct**2)))
        rows.append((params.get('wind',np.nan), params.get('dr',np.nan), params.get('Kf',np.nan), final, rmse, f))

    # write summary
    out = os.path.join(os.getcwd(),'dead_reckoning_experiments_summary.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['wind','dead_reck_sens','Kf','final_cross_m','rmse_cross_m','traj_csv'])
        for r in sorted(rows):
            w.writerow(r)

    print('Wrote summary to', out)
    # print best (lowest rmse)
    rows_sorted = sorted(rows, key=lambda x: x[4])
    print('\nTop 5 runs by RMSE:')
    for r in rows_sorted[:5]:
        print(f' wind={r[0]:.2f} dr={r[1]:.2f} Kf={r[2]:.4f} final_cross={r[3]:.2f} rmse={r[4]:.2f} file={r[5]}')

if __name__ == '__main__':
    summarize()
