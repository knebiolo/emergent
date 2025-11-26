"""
Run a small tuning sweep over ADVANCED_CONTROLLER parameters and summarize PID metrics.

This script runs headless zigzag tests (no ENC) for each combination and writes
per-run PID traces and a summary CSV at scripts/sweep_tune_summary.csv.

Adjust the 'raw_caps' and 'backcalc_betas' lists below to explore different values.
"""
import os
import itertools
import pandas as pd
from emergent.ship_abm.config import ADVANCED_CONTROLLER, PID_DEBUG, PID_TRACE, SHIP_PHYSICS
from emergent.ship_abm.simulation_core import simulation


def run_one(raw_cap_deg, backcalc_beta, out_trace_path, T=60.0):
    # configure runtime knobs
    ADVANCED_CONTROLLER['raw_cap_deg'] = raw_cap_deg
    ADVANCED_CONTROLLER['backcalc_beta'] = backcalc_beta
    # disable noisy PID debug printing during sweep
    try:
        import emergent.ship_abm.config as cfg
        cfg.PID_DEBUG = False
    except Exception:
        pass

    PID_TRACE['enabled'] = True
    PID_TRACE['path'] = out_trace_path
    # remove existing trace
    try:
        if os.path.exists(out_trace_path):
            os.remove(out_trace_path)
    except Exception:
        pass

    sim = simulation(port_name='Galveston', dt=0.1, T=T, n_agents=1, load_enc=False, test_mode='zigzag')
    sim.spawn()
    sim.run()
    return out_trace_path


def summarize_trace(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # compute metrics
    out = {}
    if 'rud_deg' in df.columns:
        out['max_rud_deg'] = df['rud_deg'].abs().max()
        out['mean_abs_rud_deg'] = df['rud_deg'].abs().mean()
    else:
        out['max_rud_deg'] = None
        out['mean_abs_rud_deg'] = None
    if 'raw_deg' in df.columns:
        out['max_raw_deg'] = df['raw_deg'].abs().max()
    else:
        out['max_raw_deg'] = None
    # saturation fraction relative to configured max rudder
    try:
        cfg_max = float(SHIP_PHYSICS.get('max_rudder', 0.0)) * 180.0 / 3.141592653589793
    except Exception:
        cfg_max = None
    if cfg_max and 'rud_deg' in df.columns:
        out['sat_frac95'] = (df['rud_deg'].abs() >= 0.95 * cfg_max).mean()
    else:
        out['sat_frac95'] = None
    # peak heading error
    out['peak_err_deg'] = df['err_deg'].abs().max() if 'err_deg' in df.columns else None
    return out


def main():
    raw_caps = [30.0, 60.0, 90.0]   # degrees
    backcalc_betas = [0.08, 0.16, 0.32]
    runs = list(itertools.product(raw_caps, backcalc_betas))

    results = []
    os.makedirs('scripts/sweep_traces', exist_ok=True)
    for raw_cap, beta in runs:
        fname = f"scripts/sweep_traces/pid_raw{int(raw_cap)}_b{int(beta*100)}.csv"
        print(f"Running raw_cap={raw_cap}°, backcalc_beta={beta} -> {fname}")
        trace_path = run_one(raw_cap, beta, fname, T=60.0)
        stats = summarize_trace(trace_path)
        row = {
            'raw_cap_deg': raw_cap,
            'backcalc_beta': beta,
            'trace': trace_path,
        }
        if stats:
            row.update(stats)
        results.append(row)

    summary = pd.DataFrame(results)
    summary_path = 'scripts/sweep_tune_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Sweep complete. Summary: {os.path.abspath(summary_path)}")


if __name__ == '__main__':
    main()
