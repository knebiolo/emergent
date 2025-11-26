#!/usr/bin/env python3
"""Run a small tuning sweep of short crosswind experiments and summarize en-route event counts.

This script temporarily patches values in emergent.ship_abm.config before constructing
`simulation` instances so we don't need to edit source config files permanently.

Output:
 - scripts/tuning_sweep_summary.csv
 - For each run, pid trace and simhist saved with run-specific suffix.

Usage: python scripts/tuning_sweep_runner.py
"""
import os, csv, shutil, importlib, time
from pathlib import Path
import numpy as np

ROOT = Path('scripts')
SUMMARY = ROOT / 'tuning_sweep_summary.csv'

# Sweep grid: (Kp_factor, Ki_factor, max_rudder_deg)
# Wider sweep grid: Kp factor, Ki factor, max_rudder_deg
GRID = []
kp_factors = [0.25, 0.5, 1.0, 2.0]
ki_factors = [0.0, 0.5, 1.0]
max_rudders = [15.0, 20.0, 25.0]
for kp in kp_factors:
    for ki in ki_factors:
        for mr in max_rudders:
            GRID.append((kp, ki, mr))

# arrival_time (for en-route filter) — reuse value from long-run experiments
ARRIVAL_TIME = 94.0

# Longer short-run to allow slower transients to emerge
# User requested 600s runs
RUN_T = 600.0  # seconds (10 minutes)

# helper to run one experiment
def run_one(run_id, kp_factor, ki_factor, max_rudder_deg):
    # import config and patch
    import emergent.ship_abm.config as C
    importlib.reload(C)
    # patch gains
    C.CONTROLLER_GAINS['Kp'] = C.CONTROLLER_GAINS['Kp'] * float(kp_factor)
    C.CONTROLLER_GAINS['Ki'] = C.CONTROLLER_GAINS['Ki'] * float(ki_factor)
    C.SHIP_PHYSICS['max_rudder'] = np.radians(float(max_rudder_deg))

    # set pid trace path unique to this run
    pid_path = ROOT / f'pid_trace_tune_run{run_id}.csv'
    C.PID_TRACE['path'] = str(pid_path)
    C.PID_TRACE['enabled'] = True

    # now import simulation (fresh)
    from emergent.ship_abm.simulation_core import simulation

    # create same short experiment as exp_waypoint_crosswind_short
    DT = 0.5
    T = RUN_T
    PORT='Rosario Strait'
    def make_const_field(u, v):
        def sampler(lons, lats, when):
            import numpy as _np
            lons = _np.atleast_1d(lons)
            N = lons.size
            out = _np.tile(_np.array([[float(u), float(v)]]), (N, 1))
            return out
        return sampler
    wind_fn = make_const_field(0.0, 6.0)
    cur_fn  = make_const_field(0.0, -0.4)

    sim = simulation(PORT, dt=DT, T=T, n_agents=1, load_enc=False, verbose=False)
    cx = 0.5 * (sim.minx + sim.maxx)
    cy = 0.5 * (sim.miny + sim.maxy)
    half_sep = 2500.0
    start = (float(cx - half_sep), float(cy))
    end   = (float(cx + half_sep), float(cy))
    sim.waypoints = [[start, end]]
    sim.wind_fn = wind_fn
    sim.current_fn = cur_fn
    sim.spawn()
    sim.run()

    # sim histories will be saved by explicit code? ensure simhist saved
    simhist_npz = ROOT / f'pid_trace_tune_run{run_id}_simhist.npz'
    try:
        # sim.history may be a list of tuples -> save with allow_pickle consumer in mind
        np.savez_compressed(simhist_npz, t=sim.t_history, psi=sim.psi_history, hd_cmd=sim.hd_cmd_history, pos=np.array(sim.history, dtype=object))
    except Exception as e:
        print('Warning: failed to save simhist for run', run_id, e)

    # correlate: use existing script correlate_pid_with_simhist.py
    corr_out = ROOT / f'pid_trace_tune_run{run_id}_correlated.csv'
    # call script as module
    import subprocess
    try:
        subprocess.run(['python','scripts/correlate_pid_with_simhist.py', str(pid_path), str(simhist_npz), '15', str(corr_out)], check=True)
        # filter en-route
        from subprocess import run
        run(['python','scripts/filter_correlated_enroute.py', str(corr_out), str(ARRIVAL_TIME), str(ROOT / f'pid_trace_tune_run{run_id}_correlated_enroute.csv')], check=True)
    except Exception as e:
        print('[SWEEP] Correlation/filter step failed for run', run_id, e)

    # read en-route count and compute mean cte pre-arrival from original pid trace? We'll reuse earlier mean field `err_deg`
    import csv
    enroute_file = ROOT / f'pid_trace_tune_run{run_id}_correlated_enroute.csv'
    count = 0
    mean_abs_err = float('nan')
    if enroute_file.exists():
        with open(enroute_file,'r',encoding='utf-8') as fh:
            rdr = csv.reader(fh)
            try:
                header = next(rdr)
            except StopIteration:
                header = None
            errs=[]
            for row in rdr:
                if not row: continue
                try:
                    errs.append(abs(float(row[2])))
                except Exception:
                    pass
            count = len(errs)
            mean_abs_err = float(np.nanmean(errs)) if errs else float('nan')

    return dict(run_id=run_id, kp_factor=kp_factor, ki_factor=ki_factor, max_rudder_deg=max_rudder_deg, enroute_event_count=count, mean_abs_err_deg=mean_abs_err)


if __name__ == '__main__':
    # stream results and append to summary file as runs finish
    header = ['run_id','Kp_factor','Ki_factor','max_rudder_deg','enroute_event_count','mean_abs_err_deg']
    # write header if file doesn't exist
    first_time = not SUMMARY.exists()
    with open(SUMMARY,'a',encoding='utf-8',newline='') as fh:
        w = csv.writer(fh)
        if first_time:
            w.writerow(header)
        for i,(kp,ki,mr) in enumerate(GRID, start=1):
            print(f'[SWEEP] Running id={i}/{len(GRID)} Kp*={kp} Ki*={ki} max_rudder={mr}deg T={RUN_T}s')
            try:
                res = run_one(i,kp,ki,mr)
            except Exception as e:
                print('[SWEEP] run_one failed for id=', i, 'err=', e)
                res = dict(run_id=i, kp_factor=kp, ki_factor=ki, max_rudder_deg=mr, enroute_event_count=0, mean_abs_err_deg=float('nan'))
            print('[SWEEP] result:', res)
            w.writerow([res['run_id'], res['kp_factor'], res['ki_factor'], res['max_rudder_deg'], res['enroute_event_count'], res['mean_abs_err_deg']])
            fh.flush()
            # short pause between runs
            time.sleep(0.5)
    print('[SWEEP] Sweep complete. Summary appended to', SUMMARY)
