"""Lightweight profiling script used by CI perf-benchmark workflow.
Generates a small pstats file at the requested output path.
"""
import argparse
import time
import random
import pstats
import cProfile
import numpy as np

from emergent.salmon_abm.tin_helpers import triangulate_and_clip


def work_simulation_step(n_points=500):
    # generate random points and values and triangulate
    xs = np.random.random(n_points) * 100.0
    ys = np.random.random(n_points) * 100.0
    pts = np.column_stack([xs, ys])
    vals = np.sin(xs * 0.1) + np.cos(ys * 0.1)
    verts, faces = triangulate_and_clip(pts, vals, poly=None, max_nodes=200)
    # do a small reduce
    if verts.shape[0] > 0:
        s = np.sum(verts[:, 2])
    else:
        s = 0.0
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=200)
    parser.add_argument('--timesteps', type=int, default=5)
    parser.add_argument('--out', type=str, default='tools/ci_profile.pstats')
    args = parser.parse_args()

    pr = cProfile.Profile()
    pr.enable()
    total = 0.0
    for t in range(max(1, args.timesteps)):
        total += work_simulation_step(n_points=max(100, args.agents))
    pr.disable()
    pr.dump_stats(args.out)
    print('profile written to', args.out)


if __name__ == '__main__':
    main()
