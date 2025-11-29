"""Run a headless HECRAS-only benchmark.

Usage: python run_headless_hecras_sim.py [--timesteps N] [--num-agents M] [--out path]

This script creates a `simulation` instance in HECRAS-only mode (no raster imports),
preloads the HECRAS KDTree, runs `environment()` for N timesteps, and writes per-timestep timings
to a CSV file specified by `--out` (defaults to `outputs/hecras_benchmark.csv`).
"""
import time
import csv
from pathlib import Path
import argparse
from emergent.salmon_abm.sockeye_SoA import simulation, load_hecras_plan_cached


parser = argparse.ArgumentParser()
parser.add_argument('--timesteps', type=int, default=200, help='Number of timesteps to run')
parser.add_argument('--num-agents', type=int, default=200, help='Number of agents')
parser.add_argument('--out', type=str, default='outputs/hecras_benchmark.csv', help='CSV output path')
parser.add_argument('--hecras_k', type=int, default=8, help='k for HECRAS IDW mapping')
args = parser.parse_args()

plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
start_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\starting_location\starting_location.shp")
long_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\Longitudinal\longitudinal.shp")
print('Using HECRAS plan:', plan)

# Minimal env_files dict placeholders
env_files = {'x_vel': '', 'y_vel': '', 'depth': '', 'wsel': '', 'elev': '', 'vel_dir': '', 'vel_mag': '', 'wetted': ''}

# Instantiate simulation in HECRAS-only mode
sim = simulation(model_dir='.', model_name='test', crs='EPSG:32604', basin='Nushagak River',
                 water_temp=8.0, start_polygon=str(start_shp), env_files=env_files,
                 longitudinal_profile=str(long_shp), fish_length=500, num_timesteps=args.timesteps,
                 num_agents=args.num_agents,
                 hecras_plan_path=str(plan), hecras_fields=['Cells Minimum Elevation', 'Water Surface',
                                                          'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y'],
                 hecras_k=args.hecras_k, use_hecras=True)

print('Preloading HECRAS map...')
start = time.time()
_m = load_hecras_plan_cached(sim, sim.hecras_plan_path, field_names=sim.hecras_fields)
print('KDTree built and fields loaded in %.3fs' % (time.time() - start))

# Ensure output directory
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

timings = []
for t in range(args.timesteps):
    t0 = time.time()
    sim.environment()
    dt = time.time() - t0
    timings.append((t, dt, float(sim.depth[0]) if hasattr(sim, 'depth') and sim.depth.size > 0 else float('nan'),
                    float(sim.x_vel[0]) if hasattr(sim, 'x_vel') and sim.x_vel.size > 0 else float('nan')))
    if t < 10 or (t + 1) % 50 == 0:
        print(f'Timestep {t}: environment() took {dt:.3f}s; sample depth[0]={timings[-1][2]:.3f}, x_vel[0]={timings[-1][3]:.3f}')

# write CSV
with out_path.open('w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['timestep', 'duration_s', 'sample_depth_0', 'sample_xvel_0'])
    writer.writerows(timings)

durations = [d for (_, d, _, _) in timings]
import statistics
print('Benchmark complete: n=%d timesteps, mean=%.6fs, median=%.6fs, stdev=%.6fs' % (
    len(durations), statistics.mean(durations), statistics.median(durations), statistics.pstdev(durations)))

print('Results written to', out_path)
