"""Headless runner for the salmon ABM to measure end-to-end step timing.

Usage:
    python tools/run_salmon_headless.py --nagents 20000 --steps 100
"""
import argparse
import time
import os
from datetime import datetime
import numpy as np
import logging
try:
    from tqdm import tqdm
except Exception:
    # lightweight fallback if tqdm not installed
    def tqdm(x, **k):
        return x

from emergent.salmon_abm.simulation import simulation


def main():
    parser = argparse.ArgumentParser(description='Run salmon ABM headless')
    parser.add_argument('--nagents', type=int, default=100, help='number of agents')
    parser.add_argument('--steps', type=int, default=100, help='number of timesteps')
    parser.add_argument('--dt', type=float, default=0.1, help='timestep (s)')
    parser.add_argument('--model_dir', type=str, default='outputs', help='model output dir')
    parser.add_argument('--basin', type=str, default='Nushagak', help='basin name')
    args = parser.parse_args()

    model_dir = args.model_dir
    model_name = 'headless_auto'
    crs = None
    basin = args.basin
    water_temp = 10.0
    start_polygon = None
    env_files = []
    longitudinal_profile = None

    # create simulation
    sim = simulation(model_dir=model_dir,
                     model_name=model_name,
                     crs=crs,
                     basin=basin,
                     water_temp=water_temp,
                     start_polygon=start_polygon,
                     env_files=env_files,
                     longitudinal_profile=longitudinal_profile,
                     num_timesteps=args.steps,
                     num_agents=args.nagents)

    # ensure noisy debug prints are disabled for timing runs
    try:
        sim.debug_behavior = False
    except Exception:
        pass
    try:
        sim.verbose = False
    except Exception:
        pass
    logging.getLogger().setLevel(logging.INFO)

    # run steps and time (with progress bar)
    t0 = time.perf_counter()
    per_step = []
    for step in tqdm(range(args.steps), desc='salmon steps'):
        s0 = time.perf_counter()
        try:
            sim.timestep(step, args.dt)
        except Exception:
            # some older code paths expect (t, dt) or (dt,)
            try:
                sim.timestep(step, 1.0)
            except Exception:
                # fallback: break
                break
        s1 = time.perf_counter()
        per_step.append(s1 - s0)
    t1 = time.perf_counter()
    wallclock = t1 - t0

    # write summary
    out_dir = os.path.join(os.getcwd(), 'outputs', 'profiling')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_csv = os.path.join(out_dir, f'salmon_headless_n{args.nagents}_s{args.steps}_{ts}.csv')
    try:
        with open(out_csv, 'w', encoding='utf-8') as fh:
            fh.write('nagents,steps,dt,wallclock_s,avg_step_s,median_step_s,std_step_s,timestamp_utc\n')
            avg = float(np.mean(per_step)) if per_step else float('nan')
            med = float(np.median(per_step)) if per_step else float('nan')
            std = float(np.std(per_step)) if per_step else float('nan')
            fh.write(f'{args.nagents},{args.steps},{args.dt},{wallclock:.9f},{avg:.9f},{med:.9f},{std:.9f},{ts}\n')
        print('Wrote', out_csv)
    except Exception as e:
        print('Failed to write summary:', e)


if __name__ == '__main__':
    main()
