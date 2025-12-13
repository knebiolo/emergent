import subprocess, sys, pathlib, shlex

def discover_tests():
    import pytest
    res = pytest.main(["--collect-only", "-q"])

if __name__ == '__main__':
    root = pathlib.Path('')
    # find pytest-discovered test files by glob
    files = sorted([str(p) for p in (root/'tests').rglob('test_*.py')])
    if not files:
        print('No test files found in tests/'); sys.exit(1)
    print(f'Found {len(files)} test files')
    lo, hi = 0, len(files)
    last_good = []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        subset = files[lo:mid]
        print(f'Trying subset {lo}:{mid} -> {len(subset)} files')
        cmd = [sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-q'] + subset
        print('Running:', ' '.join(shlex.quote(c) for c in cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print('Exit', proc.returncode)
        if proc.returncode == 0:
            # subset passed; move lo to mid
            lo = mid
            last_good = subset
        else:
            # subset failed/crashed; narrow hi to mid
            hi = mid
        # safety
        if hi - lo == 1:
            print('Narrowed to single file:', files[lo])
            sys.exit(0)
    print('Finished')
