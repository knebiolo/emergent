"""Run pytest individually per test file listed in tmp/subset_0_16.txt and save results.
Usage: python tools/run_each_test_file.py
"""
import subprocess
from pathlib import Path
import sys

subset = Path('tmp/subset_0_16.txt')
out = Path('tmp/single_run_results_0_16.txt')
out.parent.mkdir(parents=True, exist_ok=True)
if not subset.exists():
    print('subset file missing', subset)
    sys.exit(2)

lines = [l.strip() for l in subset.read_text(encoding='utf-8').splitlines() if l.strip()]
with out.open('w', encoding='utf-8') as f:
    for i, p in enumerate(lines):
        f.write(f'--- FILE {i} {p} ---\n')
        cmd = [sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q', '--basetemp', f'tmp/pytest_single_{i}', p]
        f.write('CMD: ' + ' '.join(cmd) + '\n')
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            f.write('RC: ' + str(proc.returncode) + '\n')
            f.write('STDOUT:\n')
            f.write(proc.stdout.decode('utf-8', errors='replace') + '\n')
            f.write('STDERR:\n')
            f.write(proc.stderr.decode('utf-8', errors='replace') + '\n')
        except Exception as e:
            f.write('EXC: ' + repr(e) + '\n')
        f.flush()
print('Done. Results in', out)
