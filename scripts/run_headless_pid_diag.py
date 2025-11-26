"""
Run a short headless simulation (no ENC) to collect PID traces for debugging.
Generates a CSV at scripts/pid_trace_headless_diag.csv (overwrites existing file).
"""
from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import PID_TRACE
import os
import csv
import math
import pandas as pd


def main():
    # ensure PID_TRACE writes to a specific diagnostic path
    PID_TRACE['path'] = 'scripts/pid_trace_headless_diag.csv'
    PID_TRACE['enabled'] = True

    # remove existing trace so we start fresh
    try:
        if os.path.exists(PID_TRACE['path']):
            os.remove(PID_TRACE['path'])
    except Exception:
        pass

    # create simulation in test zigzag mode so spawn() can auto-generate waypoints
    sim = simulation(port_name='Galveston', dt=0.1, T=60.0, n_agents=1, load_enc=False, test_mode='zigzag')
    # spawn will populate state/ship for test_mode
    sim.spawn()
    print("Running headless simulation for 60s to collect PID trace...")
    sim.run()
    trace_path = os.path.abspath(PID_TRACE['path'])
    print(f"Finished. PID trace at: {trace_path}")

    # Summarize rudder usage from the trace
    try:
        df = pd.read_csv(PID_TRACE['path'])
        # ensure columns exist
        if 'rud_deg' in df.columns:
            df['abs_rud'] = df['rud_deg'].abs()
            max_rud = df['abs_rud'].max()
            mean_rud = df['abs_rud'].mean()
            # configured max rudder (degrees) from SHIP_PHYSICS if available
            try:
                from emergent.ship_abm.config import SHIP_PHYSICS
                cfg_max_rud_deg = float(SHIP_PHYSICS.get('max_rudder', 0.0)) * 180.0 / 3.141592653589793
            except Exception:
                cfg_max_rud_deg = None
            # fraction of time at or above 95% of configured max rudder (if known)
            if cfg_max_rud_deg is not None and cfg_max_rud_deg > 0:
                sat_frac_cfg95 = (df['abs_rud'] >= 0.95 * cfg_max_rud_deg).mean()
                sat_frac_cfg99 = (df['abs_rud'] >= 0.99 * cfg_max_rud_deg).mean()
            else:
                sat_frac_cfg95 = None
                sat_frac_cfg99 = None
            # raw PID pre-saturation extremes
            raw_max = None
            if 'raw_deg' in df.columns:
                raw_max = df['raw_deg'].abs().max()
            print(f"PID trace summary: max_rudder_deg={max_rud:.2f}, mean_abs_rudder_deg={mean_rud:.2f}, raw_max_deg={raw_max if raw_max is not None else 'NA'}")
            if sat_frac_cfg95 is not None:
                print(f"Fraction of time >=95% cfg_max_rudder: {sat_frac_cfg95:.3f}, >=99%: {sat_frac_cfg99:.3f} (cfg_max={cfg_max_rud_deg:.2f}deg)")
        else:
            print("PID trace written but 'rud_deg' column not found; raw columns: ", df.columns.tolist())
    except Exception as e:
        print(f"Could not summarize PID trace: {e}")


if __name__ == '__main__':
    main()
