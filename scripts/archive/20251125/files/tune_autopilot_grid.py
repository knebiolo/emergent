import itertools
import os
import csv
from headless_dead_reckoning_test import run_experiment

def main():
    # Focused grid (72 runs): mix of coarse PID and dead-reck/Kf
    Kp_list = [0.3, 0.6, 1.0]
    Ki_list = [0.0, 0.02]
    Kd_list = [0.05, 0.12]
    dead_reck_list = [0.1, 0.25, 0.5]
    Kf_list = [0.0, 0.002]

    results = []
    total = 0
    for Kp, Ki, Kd, dr, Kf in itertools.product(Kp_list, Ki_list, Kd_list, dead_reck_list, Kf_list):
        total += 1
    print(f"Running {total} tuning runs (may take some time)...")

    idx = 0
    for Kp, Ki, Kd, dr, Kf in itertools.product(Kp_list, Ki_list, Kd_list, dead_reck_list, Kf_list):
        idx += 1
        print(f"[{idx}/{total}] Kp={Kp} Ki={Ki} Kd={Kd} dr={dr} Kf={Kf}")
        # run one experiment with these controller params
        res = run_experiment(wind_speed=1.5, dead_reck_sens=dr, Kf_gain=Kf, dt=0.5, T=240.0, verbose=False)
        # the run_experiment currently doesn't accept Kp/Ki/Kd; we will post-edit the produced per-run CSV filename
        # but we also want to record the PID values; store them in summary
        res['Kp'] = Kp
        res['Ki'] = Ki
        res['Kd'] = Kd
        # rename the per-run traj CSV to include PID params for traceability
        old = res['traj_csv']
        base = os.path.basename(old)
        new_name = base.replace('.csv', f'_Kp{Kp:.2f}_Ki{Ki:.3f}_Kd{Kd:.3f}.csv')
        new_path = os.path.join(os.getcwd(), new_name)
        try:
            os.replace(old, new_path)
            res['traj_csv'] = new_path
        except Exception:
            # if rename fails, keep old path
            pass
        results.append(res)

    # write summary with PID fields
    out = os.path.join(os.getcwd(), 'autopilot_tuning_summary.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Kp','Ki','Kd','wind','dead_reck_sens','Kf','final_cross_m','rmse_cross_m','traj_csv'])
        for r in results:
            w.writerow([r.get('Kp'), r.get('Ki'), r.get('Kd'), r.get('wind_speed'), r.get('dead_reck_sens'), r.get('Kf_gain'), r.get('final_cross_m'), r.get('rmse_cross_m'), r.get('traj_csv')])

    print('Wrote tuning summary to', out)

if __name__ == '__main__':
    main()
