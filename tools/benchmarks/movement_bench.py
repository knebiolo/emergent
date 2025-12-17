"""Microbenchmark script to compare numba vs numpy implementations of swim_core and drag kernels."""
import time
import numpy as np
from emergent.fish_passage.movement import swim_core, drag_and_battery

def bench(fn, *args, repeats=100):
    t0 = time.time()
    for _ in range(repeats):
        fn(*args)
    t1 = time.time()
    return (t1 - t0) / repeats

def run_bench():
    N = 1000
    positions = np.zeros((N,2), dtype=float)
    headings = np.zeros(N, dtype=float)
    speeds = np.ones(N, dtype=float)
    env = np.zeros((N,2), dtype=float)
    s_time = bench(swim_core, positions, headings, speeds, env, 1.0, repeats=20)
    d_time = bench(drag_and_battery, positions, speeds, headings, env, 1.0, repeats=20)
    print(f"swim_core avg: {s_time:.6f}s, drag_and_battery avg: {d_time:.6f}s")

if __name__ == '__main__':
    run_bench()
