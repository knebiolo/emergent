"""High-fidelity PRBS identification wrapper.
Calls the existing PRBS-with-servo harness with smaller dt and longer duration,
then runs the PRBS analysis script to compute ETFE/coherence and ARX fits.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

# Load run_prbs_with_servo from scripts/id_prbs_with_servo.py
spec = importlib.util.spec_from_file_location('id_prbs_with_servo', str(ROOT / 'id_prbs_with_servo.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_prbs_with_servo = getattr(mod, 'run_prbs_with_servo')

def main():
    # high fidelity params
    dt = 0.1
    T = 300.0
    bit_time = 0.5
    amp_deg = 6.0
    U = 5.0
    print('Running high-fidelity servo PRBS: dt=', dt, 'T=', T, 'bit_time=', bit_time)
    out_csv, out_json = run_prbs_with_servo(U_nom=U, amp_deg=amp_deg, dt=dt, T=T, bit_time=bit_time, out_prefix='id_servo_hifi')
    print('PRBS telemetry written to', out_csv)

    # run analysis pipeline
    print('Running PRBS analysis pipeline...')
    # call the existing prbs_analysis_and_id.py using same Python
    analysis = str(ROOT / 'prbs_analysis_and_id.py')
    subprocess.check_call([sys.executable, analysis])
    print('PRBS analysis complete. Check figs/prbs_id/')

if __name__ == '__main__':
    main()
