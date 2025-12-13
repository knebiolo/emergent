#!/usr/bin/env python3
import subprocess, sys, pathlib, shlex

root = pathlib.Path('.')
# discover pytest test files only in repository folders to avoid virtualenv/site-packages tests
search_dirs = [root / 'tests', root / 'src', root / 'tools']
files_set = set()
for d in search_dirs:
    if d.exists():
        for p in d.rglob('test_*.py'):
            files_set.add(str(p))
files = sorted(files_set)
print('Discovered', len(files), 'test files')
if not files:
    sys.exit(1)

lo, hi = 0, len(files)
# binary search narrowing
while lo < hi:
    if hi - lo == 1:
        print('\nOffending file:', files[lo])
        sys.exit(0)
    mid = (lo + hi) // 2
    subset = files[lo:mid]
    print(f'\nTrying range {lo}:{mid} -> {len(subset)} files')
    # write subset to a temp file and invoke pytest_runner to avoid long command lines on Windows
    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt') as tf:
        tf_name = tf.name
        tf.write('\n'.join(subset))
    cmd = [sys.executable, '-X', 'faulthandler', 'tools/pytest_runner.py', tf_name]
    print('Running via runner:', ' '.join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode('utf-8', errors='replace')
    print('Exit code', proc.returncode)
    print('--- output (first 2000 chars) ---')
    print(out[:2000])
    print('--- end output ---')
    if proc.returncode == 0:
        lo = mid
    else:
        hi = mid

print('No single offending file found')
sys.exit(2)
