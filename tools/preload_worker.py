"""Start a preloaded worker process that imports simulation modules and performs warmup.

Usage: python tools/preload_worker.py [--port 0]

This script is interactive: it warms numba/llvmlite and then sleeps, ready to accept work
or be used to run the benchmark inside the same process to avoid JIT noise.
"""
import sys
import time
from importlib import import_module

if __name__ == '__main__':
    print('preload_worker: importing emergent modules and running warmup')
    try:
        import tools.numba_warmup as nw
    except Exception:
        # fallback import path
        try:
            nw = import_module('tools.numba_warmup')
        except Exception:
            nw = None
    try:
        # also import core sim module
        simmod = import_module('emergent.salmon_abm.sockeye_SoA')
    except Exception:
        simmod = None
    try:
        if nw is not None and hasattr(nw, 'main'):
            nw.main()
    except Exception as e:
        print('preload warmup failed:', e)
    print('preload_worker: warmup complete; sleeping. Use this process to run benchmarks to avoid JIT overhead.')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print('preload_worker: exiting')
