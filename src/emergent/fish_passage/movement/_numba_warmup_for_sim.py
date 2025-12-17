"""Helper to warm up numba kernels so first-tick JIT overhead is outside timed runs."""
def warmup_swim_core(swim_core_func):
    # call once with small arrays to trigger compilation
    import numpy as np
    positions = np.zeros((2,2), dtype=float)
    headings = np.zeros(2, dtype=float)
    speeds = np.zeros(2, dtype=float)
    env = np.zeros((2,2), dtype=float)
    swim_core_func(positions, headings, speeds, env, 1.0)

def warmup_drag_and_battery(drag_and_battery_func):
    import numpy as np
    positions = np.zeros((2,2), dtype=float)
    speeds = np.zeros(2, dtype=float)
    headings = np.zeros(2, dtype=float)
    env = np.zeros((2,2), dtype=float)
    drag_and_battery_func(positions, speeds, headings, env, 1.0)
