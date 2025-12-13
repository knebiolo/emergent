import os
import sys
import subprocess
from pathlib import Path


def main():
    env = os.environ.copy()
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')

    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    basetemp = tmp / 'pytest_basetemp'
    basetemp.mkdir(exist_ok=True)

    args = [sys.executable, '-X', 'faulthandler', '-m', 'pytest', f'--basetemp={basetemp}', '-q']
    print('Running:', ' '.join(args))
    rc = subprocess.call(args, env=env)
    print('Exit code:', rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
