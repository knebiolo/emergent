"""Microbenchmark for emergent.fish_passage.metrics

Usage:
    python scripts/bench_metrics.py --n 100 --repeats 5 --numba

Options:
    --n: agent count (default 100)
    --repeats: repetitions per test
    --numba: precompile Numba kernels before timing
"""
import time
import argparse
import numpy as np

from emergent.fish_passage import metrics


def make_random_state(n, seed=0):
    rng = np.random.RandomState(seed)
    positions = rng.uniform(-10.0, 10.0, size=(n, 2)).astype(np.float64)
    headings = rng.uniform(-np.pi, np.pi, size=(n,)).astype(np.float64)
    velocities = rng.uniform(0.0, 2.0, size=(n,)).astype(np.float64)
    body_lengths = rng.uniform(0.5, 2.0, size=(n,)).astype(np.float64)
    class BW:
        pass
    bw = BW()
    return positions, headings, velocities, body_lengths, bw


def time_fn(fn, args, repeats=5):
    # warmup
    fn(*args)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return min(ts), sum(ts) / len(ts)


def run_bench(n=100, repeats=5, precompile_numba=False):
    pos, headings, velocities, bl, bw = make_random_state(n)
    if precompile_numba and getattr(metrics, '_HAS_NUMBA', False):
        print('Precompiling numba kernels...')
        metrics.compile_numba_kernels()

    print(f'Agent count: {n}, repeats: {repeats}, numba available: {getattr(metrics, "_HAS_NUMBA", False)}')

    # schooling (numpy)
    tmin, tavg = time_fn(lambda p, h, b, bw_: metrics.compute_schooling_metrics_biological(p, h, b, bw_), (pos, headings, bl, bw), repeats)
    print(f'schooling (numpy/numba path): min={tmin:.6f}s avg={tavg:.6f}s')

    # drafting
    tmin, tavg = time_fn(lambda p, h, v, b, bw_: metrics.compute_drafting_benefits(p, h, v, b, bw_), (pos, headings, velocities, bl, bw), repeats)
    print(f'drafting (numpy/numba path): min={tmin:.6f}s avg={tavg:.6f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--numba', action='store_true')
    args = parser.parse_args()
    run_bench(args.n, args.repeats, precompile_numba=args.numba)


if __name__ == '__main__':
    main()
