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
    # If a helper exists, call it — but first do an aggressive preload
    try:
        # Pre-import numba/llvmlite to force library deserialization outside the timed run
        try:
            import numba
            # llvmlite import triggers JIT backend initialization
            import llvmlite.binding as llvm
            # touch LLVM to ensure binding initialized
            _ = llvm.get_host_cpu_name()
        except Exception:
            pass

        # find any Simulation class and call _numba_warmup_for_sim if present
        for name in dir(simmod):
            obj = getattr(simmod, name)
            if hasattr(obj, '__init__') and hasattr(simmod, '_numba_warmup_for_sim'):
                print('calling _numba_warmup_for_sim (aggressive)')
                # create a dummy object with production-shaped arrays
                class _DummySim:
                    pass
                d = _DummySim()
                # choose a realistic large agent count to force the same specializations
                d.num_agents = 2000
                import numpy as np
                # choose swim buffer width matching typical runs; fall back to 4
                d.swim_speeds = np.zeros((d.num_agents, 8), dtype=np.float64)
                # call sim-level warmup if present
                try:
                    simmod._numba_warmup_for_sim(d)
                except Exception as e:
                    print('sim warmup helper failed:', e)

                # additionally call exact hot kernels/wrappers directly if available
                try:
                    # prepare exact-shape contiguous arrays for kernel signatures
                    ones = np.ones(d.num_agents, dtype=np.float64)
                    zeros = np.zeros(d.num_agents, dtype=np.float64)
                    bmask = np.ones(d.num_agents, dtype=np.bool_)
                    bi = np.zeros(d.num_agents, dtype=np.int64)
                    swim_buf = np.zeros((d.num_agents, d.swim_speeds.shape[1]), dtype=np.float64)
                    # call known-numba kernels with exact signatures used in production
                    try:
                        if hasattr(simmod, '_compute_drags_numba'):
                            simmod._compute_drags_numba(ones, ones, ones, ones, bmask, 1.0, ones, ones, ones, bi)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_swim_speeds_numba'):
                            simmod._swim_speeds_numba(ones, ones, ones, ones)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_assess_fatigue_core'):
                            # function expects many arrays; call with ones/zeros and swim buffer
                            simmod._assess_fatigue_core(ones, ones, ones, ones, ones, ones, ones, swim_buf)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_merged_swim_drag_fatigue_numba'):
                            simmod._merged_swim_drag_fatigue_numba(ones, ones, ones, ones, bmask, 1.0, ones, ones, 0.0, None, ones, ones, ones, swim_buf)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_drag_and_battery_numba'):
                            # note: last arg update_battery True/False toggles battery updates
                            simmod._drag_and_battery_numba(ones, ones, ones, ones, bmask, 1.0, ones, ones, 0.0, None, ones, ones, swim_buf, True)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_wrap_drag_fun_numba'):
                            simmod._wrap_drag_fun_numba(zeros, zeros, zeros, zeros, bmask, 1.0, ones, ones, 0.0, None, None)
                    except Exception:
                        pass
                    try:
                        if hasattr(simmod, '_wrap_merged_battery_numba'):
                            simmod._wrap_merged_battery_numba(ones, ones, ones, bmask, 0.1)
                    except Exception:
                        pass
                except Exception:
                    pass
                break
    except Exception as e:
        print('warmup failed', e)

    # Try to force llvmlite finalization hooks; best-effort and optional
    try:
        import llvmlite.binding as llvm
        # finalize any pending modules/objects
        if hasattr(llvm, 'finalize'):
            try:
                llvm.finalize()
            except Exception:
                pass
    except Exception:
        pass

if __name__ == '__main__':
    main()
