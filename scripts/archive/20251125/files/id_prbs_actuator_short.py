"""Short actuator-focused PRBS runs to identify servo/actuator dynamics.
Calls run_prbs with faster bit times and shorter duration.
"""
from datetime import datetime
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('id_prbs', str(ROOT / 'scripts' / 'id_prbs.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_prbs = getattr(mod, 'run_prbs')

def main():
    speeds = [3.0, 5.0, 7.0]
    dt = 0.2
    T = 180.0
    bit_time = 1.0
    amp_deg = 6.0
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    print(f"Starting actuator-focused PRBS batch at {ts}: dt={dt}s, T={T}s, bit_time={bit_time}s")
    for U in speeds:
        try:
            print(f"Running actuator PRBS for U={U} m/s")
            out_csv, out_json = run_prbs(U_nom=U, amp_deg=amp_deg, dt=dt, T=T, bit_time=bit_time, out_prefix=f'id_actuator_short')
            print('Wrote:', out_csv, out_json)
        except Exception as e:
            print(f"PRBS run failed for U={U}: {e}")

if __name__ == '__main__':
    main()
