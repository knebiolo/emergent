#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print('Usage: pytest_runner.py <subset_file>')
        return 2
    subset_file = Path(sys.argv[1])
    if not subset_file.exists():
        print('Subset file not found:', subset_file)
        return 2
    paths = [line.strip() for line in subset_file.read_text().splitlines() if line.strip()]
    if not paths:
        print('No paths in subset file')
        return 0
    import pytest, os, uuid
    # create a per-run basetemp inside repo tmp/ to avoid using OS temp
    basetemp_root = Path('tmp')
    basetemp_root.mkdir(exist_ok=True)
    unique = f'pytest_run_{os.getpid()}_{uuid.uuid4().hex[:8]}'
    basetemp = basetemp_root / unique
    basetemp.mkdir(exist_ok=True)
    # Run pytest in a fresh process context (we're already in a spawned process)
    args = ['-q', '--basetemp', str(basetemp)] + paths
    # Limit BLAS and OMP thread counts in this process as well
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    return pytest.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
