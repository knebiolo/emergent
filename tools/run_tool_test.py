import os
import sys
import subprocess
from pathlib import Path

def run(test_path):
    env = os.environ.copy()
    env['QT_QPA_PLATFORM_PLUGIN_PATH'] = r"C:\Users\Kevin.Nebiolo\.conda\envs\emergent\Library\plugins\platforms"
    env['QT_QPA_PLATFORM'] = 'minimal'
    env['OPENBLAS_NUM_THREADS'] = env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'

    proc = subprocess.run([sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q', test_path], env=env, capture_output=True)

    out_dir = Path('tmp')
    out_dir.mkdir(exist_ok=True)
    name = Path(test_path).stem
    out_file = out_dir / f"{name}_out.txt"
    rc_file = out_dir / f"{name}_rc.txt"

    with open(out_file, 'wb') as f:
        f.write(proc.stdout)
        f.write(proc.stderr)

    with open(rc_file, 'w') as f:
        f.write(str(proc.returncode))

    print(f"Wrote {out_file} and {rc_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: run_tool_test.py <test_path>')
        sys.exit(2)
    run(sys.argv[1])
