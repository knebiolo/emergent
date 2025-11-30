"""Script to run Numba warmup for the emergent simulation.

Usage: python tools/numba_warmup.py
"""
import os
import sys
from importlib import import_module

def main():
    # Import the simulation module and construct a minimal simulation to warm numba
    try:
        simmod = import_module('emergent.salmon_abm.sockeye_SoA')
    except Exception:
        print('failed to import sockeye_SoA')
        sys.exit(1)
    # If a helper exists, call it
    try:
        # find any Simulation class and call _numba_warmup_for_sim if present
        for name in dir(simmod):
            obj = getattr(simmod, name)
            if hasattr(obj, '__init__') and hasattr(simmod, '_numba_warmup_for_sim'):
                print('calling _numba_warmup_for_sim')
                # create a minimal dummy object with num_agents attribute
                class _DummySim:
                    pass
                d = _DummySim()
                d.num_agents = 128
                import numpy as np
                d.swim_speeds = np.zeros((128, 4))
                try:
                    simmod._numba_warmup_for_sim(d)
                except Exception as e:
                    print('warmup helper failed:', e)
                break
    except Exception as e:
        print('warmup failed', e)

if __name__ == '__main__':
    main()
