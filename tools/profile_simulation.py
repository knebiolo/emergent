"""Profiling harness for emergent ABM simulation.

This script runs a short simulation loop or a targeted hotspot loop and
profiles CPU time using timeit and cProfile. It writes a small report to
outputs/profiling.

Usage:
    python tools/profile_simulation.py --mode=hotspot --iters=100 --nagents=1000
    python tools/profile_simulation.py --mode=sim --steps=10 --config=scenarios/dali_test_quick.json

"""
from __future__ import annotations
import argparse
import os
import time
import cProfile
import pstats
import io

import numpy as np

from emergent.salmon_abm.behavior import behavior
from emergent.salmon_abm import simulation as simmod
import os


def profile_hotspot(n_agents=1000, iters=100, *, avoid_mode: str = "dense", history_len: int = 1024, seed_k: int = 64):
    # Create a minimal mock simulation object with essentials used by behavior
    class MockSim:
        def __init__(self, n):
            self.num_agents = n
            self.n_agents = n
            self.X = np.random.rand(n) * 1000.0
            self.Y = np.random.rand(n) * 1000.0
            self.heading = np.zeros(n)
            self.sog = np.full(n, 0.2)
            self.length = 100.0
            self.mental_map_transform = (1, 0, 0, 0, 1, 0)
            self.depth_rast_transform = self.mental_map_transform
            self.vel_mag_rast_transform = self.mental_map_transform
            self.num_agents = n
            # minimal arrays
            self.battery = np.ones(n)
            self.recover_stopwatch = np.zeros(n)
            self.swim_behav = np.ones(n, dtype=np.int32)
            self.ideal_sog = np.full(n, 0.5)
            # Avoid-memory config
            self.use_sparse_avoid_memory = (str(avoid_mode).lower() == "sparse")
            self.avoid_memory_horizon_s = 7200.0
            self.avoid_history_chunk = 32
            # simple in-memory HDF5-like mock: dict of arrays
            self._h5 = {
                'memory/0': np.zeros((200, 200), dtype=float),
                'x_coords': np.zeros((200, 200), dtype=float),
                'y_coords': np.zeros((200, 200), dtype=float),
                'environment/depth': np.zeros((200, 200), dtype=float),
                'environment/vel_mag': np.zeros((200, 200), dtype=float),
            }
            self.model_dir = os.path.join('outputs', 'profiling')
            os.makedirs(self.model_dir, exist_ok=True)
            # optional override for behavior batch size (tuning)
            bsize = os.environ.get('BEHAVIOR_BATCH_SIZE')
            if bsize is not None:
                try:
                    self.behavior_batch_size = int(bsize)
                except Exception:
                    pass
            # Optional: seed sparse avoid history so behavior takes the sparse path.
            if self.use_sparse_avoid_memory:
                k = max(1, int(history_len))
                self.avoid_hist_rows = np.full((n, k), -1, dtype=np.int16)
                self.avoid_hist_cols = np.full((n, k), -1, dtype=np.int16)
                self.avoid_hist_t = np.full((n, k), np.nan, dtype=np.float32)
                self.avoid_hist_pos = np.zeros((n,), dtype=np.int32)
                # seed a handful of visited points for each agent (kept small for setup speed)
                sk = max(1, min(int(seed_k), k))
                rr = np.random.randint(0, 200, size=(n, sk), dtype=np.int16)
                cc = np.random.randint(0, 200, size=(n, sk), dtype=np.int16)
                # spread timestamps so some fall within the active horizon window
                base_t = 1000.0
                tt = (base_t - np.random.uniform(20.0, 2000.0, size=(n, sk))).astype(np.float32)
                self.avoid_hist_rows[:, :sk] = rr
                self.avoid_hist_cols[:, :sk] = cc
                self.avoid_hist_t[:, :sk] = tt
                self.avoid_hist_pos[:] = sk % k

        def sample_environment(self, transform, key):
            # simple constant sample
            return np.full(self.num_agents, 0.1)

    sim = MockSim(n_agents)
    bh = behavior(dt=1.0, simulation_object=sim)

    # Numba warmup: try to invoke the batched repulsive kernel once with a tiny synthetic batch
    # to ensure JIT compilation happens before timed runs. If Numba is unavailable or the
    # kernel wrapper isn't present, fall back to calling the Python warmup loop.
    try:
        # create a tiny synthetic batch via behavior internals if available
        if hasattr(bh, '_repulsive_batched_core_safe'):
            # prepare tiny buffers (1 agent, 1 pixel) and call safe wrapper
            # Many internals are private; we attempt a light-weight call that should JIT the kernel.
            try:
                bh._repulsive_batched_core_safe(np.zeros(1, dtype=np.float32),
                                                np.zeros(1, dtype=np.float32),
                                                np.ones(1, dtype=np.float32),
                                                np.array([0], dtype=np.int32),
                                                np.array([1], dtype=np.int32),
                                                np.array([0], dtype=np.int32),
                                                np.array([1], dtype=np.int32),
                                                1)
            except Exception:
                # fall back to Python warmup below
                for _ in range(3):
                    bh.already_been_here(weight=100.0, t=0.0)
        else:
            for _ in range(3):
                bh.already_been_here(weight=100.0, t=0.0)
    except Exception:
        for _ in range(3):
            bh.already_been_here(weight=100.0, t=0.0)

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    for i in range(iters):
        bh.already_been_here(weight=100.0, t=float(i))
    t1 = time.perf_counter()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    profile_text = s.getvalue()

    summary = f"iters={iters}, n_agents={n_agents}, wallclock={t1-t0:.3f}s\n"
    outdir = 'outputs/profiling'
    os.makedirs(outdir, exist_ok=True)
    tag = os.environ.get('RUN_TAG')
    mode_tag = str(avoid_mode).lower()
    fname = f'hotspot_profile_{mode_tag}_{tag}.txt' if tag else f'hotspot_profile_{mode_tag}.txt'
    with open(os.path.join(outdir, fname), 'w', encoding='utf-8') as fh:
        fh.write(summary)
        fh.write(profile_text)

    print(f'Wrote profiling output to outputs/profiling/{fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('hotspot', 'sim'), default='hotspot')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--nagents', type=int, default=500)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--debug-behavior', action='store_true',
                        help='Enable debug_behavior on the mock simulation to force non-empty windows and richer batch logs')
    parser.add_argument('--avoid-mode', choices=('dense', 'sparse'), default='sparse',
                        help='Which avoid-memory implementation to profile')
    parser.add_argument('--avoid-history-len', type=int, default=1024,
                        help='Sparse history length per agent (only used when --avoid-mode=sparse)')
    parser.add_argument('--avoid-seed-k', type=int, default=64,
                        help='How many history entries to seed per agent (only used when --avoid-mode=sparse)')
    parser.add_argument('--tag', type=str, default=None, help='Optional run tag to annotate output files')
    args = parser.parse_args()

    if args.mode == 'hotspot':
        # export debug flag to environment for behavior to pick up if needed
        if args.debug_behavior:
            os.environ['DEBUG_BEHAVIOR'] = '1'
        if args.tag:
            os.environ['RUN_TAG'] = args.tag
        profile_hotspot(
            n_agents=args.nagents,
            iters=args.iters,
            avoid_mode=str(args.avoid_mode),
            history_len=int(args.avoid_history_len),
            seed_k=int(args.avoid_seed_k),
        )
    else:
        print('sim mode not implemented in this harness yet')
