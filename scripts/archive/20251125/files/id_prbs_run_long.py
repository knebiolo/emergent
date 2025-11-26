"""Wrapper to run longer, lower-frequency PRBS experiments for system ID.

This script calls `run_prbs` from `scripts/id_prbs.py` with conservative
low-frequency settings to better excite slow yaw dynamics:
 - dt = 0.5 s
 - T  = 1200 s (20 minutes per run)
 - bit_time = 10 s

Run as: python scripts/id_prbs_run_long.py
"""
from datetime import datetime
import traceback
import importlib.util
from pathlib import Path

# dynamic import of scripts/id_prbs.py so this wrapper can be run from repo root
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('id_prbs', str(ROOT / 'scripts' / 'id_prbs.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_prbs = getattr(mod, 'run_prbs')

def main():
    speeds = [3.0, 5.0, 7.0]
    dt = 0.5
    T = 1200.0
    bit_time = 10.0
    amp_deg = 4.0
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    print(f"Starting long PRBS batch at {ts}: dt={dt}s, T={T}s, bit_time={bit_time}s")
    for U in speeds:
        try:
            print(f"Running PRBS for U={U} m/s")
            out_csv, out_json = run_prbs(U_nom=U, amp_deg=amp_deg, dt=dt, T=T, bit_time=bit_time, out_prefix=f'id_long')
            print('Wrote:', out_csv, out_json)
        except Exception as e:
            print(f"PRBS run failed for U={U}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()
