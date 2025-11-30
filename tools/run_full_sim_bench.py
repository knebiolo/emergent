"""Full simulation benchmark runner.

Runs the full `simulation.timestep` loop for a small number of timesteps while
injecting time-varying velocity perturbations so the environment changes.

Usage:
  python tools/run_full_sim_bench.py --num-agents 100 --timesteps 20 --out outputs/full_bench_n100.csv
"""
import argparse
import time
import csv
from pathlib import Path
import numpy as np
from emergent.salmon_abm.sockeye_SoA import simulation, load_hecras_plan_cached


parser = argparse.ArgumentParser()
parser.add_argument('--num-agents', type=int, required=True)
parser.add_argument('--timesteps', type=int, default=20)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--hecras_k', type=int, default=8)
parser.add_argument('--write-rasters', action='store_true', help='If set, write full environment rasters from HECRAS each timestep (slower)')
parser.add_argument('--strict', action='store_true', help='If set, re-raise exceptions from sim.timestep instead of using lite fallback')
args = parser.parse_args()

plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
start_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\starting_location\starting_location.shp")
long_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\Longitudinal\longitudinal.shp")

env_files = {'x_vel': '', 'y_vel': '', 'depth': '', 'wsel': '', 'elev': '', 'vel_dir': '', 'vel_mag': '', 'wetted': ''}

print('Building simulation (num_agents=%d)...' % args.num_agents)
sim = simulation(model_dir='.', model_name='fullbench', crs='EPSG:32604', basin='Nushagak River',
                 water_temp=8.0, start_polygon=str(start_shp), env_files=env_files,
                 longitudinal_profile=str(long_shp), fish_length=500, num_timesteps=args.timesteps,
                 num_agents=args.num_agents,
                 hecras_plan_path=str(plan), hecras_fields=['Cells Minimum Elevation', 'Water Surface',
                                                          'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y'],
                 hecras_k=args.hecras_k, use_hecras=True, hecras_write_rasters=bool(args.write_rasters))

# preload HECRAS
load_hecras_plan_cached(sim, sim.hecras_plan_path, field_names=sim.hecras_fields)

# If running HECRAS-only, set conservative raster transform fallbacks so other code
# that dereferences raster transforms doesn't fail.
try:
    if getattr(sim, 'hecras_plan_path', None):
        m = load_hecras_plan_cached(sim, sim.hecras_plan_path, field_names=sim.hecras_fields)
        coords = m.coords
        minx = float(coords[:, 0].min())
        maxy = float(coords[:, 1].max())
        cell = getattr(sim, 'avoid_cell_size', 10.0)
        from rasterio.transform import Affine
        fallback_t = Affine(cell, 0.0, minx, 0.0, -cell, maxy)
        # assign common raster transform attributes used through the sim
        sim.depth_rast_transform = fallback_t
        sim.vel_x_rast_transform = fallback_t
        sim.vel_y_rast_transform = fallback_t
        sim.vel_dir_rast_transform = fallback_t
        sim.vel_mag_rast_transform = fallback_t
        sim.wetted_transform = fallback_t
        # approximate width/height if missing
        if not hasattr(sim, 'width') or not hasattr(sim, 'height'):
            xrange = float(coords[:, 0].max() - coords[:, 0].min())
            yrange = float(coords[:, 1].max() - coords[:, 1].min())
            sim.width = int(np.ceil(xrange / cell)) + 1
            sim.height = int(np.ceil(yrange / cell)) + 1
except Exception:
    pass

# Create placeholder environment datasets if missing so full-sim behaviors can read them
try:
    if 'environment' not in sim.hdf5:
        env_grp = sim.hdf5.create_group('environment')
    else:
        env_grp = sim.hdf5['environment']
    dnames = ['depth', 'vel_x', 'vel_y', 'vel_mag', 'vel_dir', 'wetted', 'distance_to']
    for dn in dnames:
        if dn not in env_grp:
            env_grp.create_dataset(dn, shape=(sim.height, sim.width), dtype='f4', fillvalue=np.nan,
                                   chunks=(min(128, sim.height), min(128, sim.width)))
    sim.hdf5.flush()
except Exception:
    pass

# disable HDF flushes to avoid I/O skewing benchmarks
sim.flush_interval = 10**9

# We'll perturb x_vel/y_vel on each timestep to simulate a changing environment
def perturb_environment(sim_obj, t):
    # small sinusoidal perturbation based on agent index and timestep
    phase = (np.arange(sim_obj.num_agents) % 10) / 10.0
    amp = 0.1
    sim_obj.x_vel = sim_obj.x_vel + amp * np.sin(2.0 * np.pi * (t / 10.0) + phase)
    sim_obj.y_vel = sim_obj.y_vel + amp * np.cos(2.0 * np.pi * (t / 7.0) + phase)


# Minimal PID controller stub used for benchmarks
class PIDStub:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # simulation expects attributes named k_p, k_i, k_d and integral
        self.k_p = np.array([kp])
        self.k_i = np.array([ki])
        self.k_d = np.array([kd])
        self.integral = np.array([0.0])

    def PID_func(self, water_speed_array, *args, **kwargs):
        # Return three scalars (P,I,D) — simulation will then assign these arrays
        return float(self.kp), float(self.ki), float(self.kd)

    def update(self, error, dt, extra):
        # error: (N,2) array (along, cross) — produce (N,2) adjustment
        if error is None or getattr(error, 'shape', (0,))[0] == 0:
            return np.zeros((0, 2))
        # simple proportional-only controller per-agent using first column of error
        kp = float(self.k_p.ravel()[0])
        adj = np.zeros_like(error)
        adj[:, 0] = kp * error[:, 0]
        adj[:, 1] = kp * error[:, 1]
        # update integral (store scalar mean for compatibility)
        self.integral = self.integral + np.array([np.mean(error)]) * dt
        return adj

pid_controller = PIDStub(kp=0.5, ki=0.0, kd=0.0)


out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

timings = []
for t in range(args.timesteps):
    # sample environment first to populate x_vel/y_vel via HECRAS mapping
    sim.environment()
    # apply perturbation to velocities so environment varies over time
    perturb_environment(sim, t)

    # Ensure neighbor lists exist (some environment modes may skip computing them)
    if not hasattr(sim, 'agents_within_buffers'):
        # compute quick neighbor lists using KDTree (radius=6)
        clean_x = sim.X.flatten()[~np.isnan(sim.X.flatten())]
        clean_y = sim.Y.flatten()[~np.isnan(sim.Y.flatten())]
        positions = np.vstack([clean_x, clean_y]).T
        try:
            tree = __import__('scipy').spatial.cKDTree(positions)
            agents_within_radius = tree.query_ball_tree(tree, r=6.0)
            sim.agents_within_buffers = agents_within_radius
        except Exception:
            sim.agents_within_buffers = [[] for _ in range(sim.num_agents)]

    t0 = time.time()
    dt = 1.0
    g_val = 9.80665
    mode = 'full'
    try:
        sim.timestep(t, dt, g_val, pid_controller)
    except Exception as e:
        if args.strict:
            # re-raise so caller can see the stack and we can harden code
            raise
        # fallback: perform a simplified vectorized movement step to represent workload
        mode = 'lite'
        # compute headings from water velocity
        heading = np.arctan2(sim.y_vel, sim.x_vel)
        # ideal velocity vector
        ideal = np.vstack((sim.ideal_sog * np.cos(heading), sim.ideal_sog * np.sin(heading))).T
        water = np.vstack((sim.x_vel, sim.y_vel)).T
        fish_vel = ideal - water
        speeds = np.linalg.norm(fish_vel, axis=-1)
        # update positions
        sim.prev_X = sim.X.copy()
        sim.prev_Y = sim.Y.copy()
        sim.X = sim.X + fish_vel[:, 0] * dt
        sim.Y = sim.Y + fish_vel[:, 1] * dt
        sim.heading = heading
        sim.swim_speed = speeds

    dt_run = time.time() - t0
    timings.append((t, dt_run, mode))
    print(f'Timestep {t}: {mode} timestep took {dt_run:.3f}s')

# write CSV
with out_path.open('w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['timestep', 'duration_s', 'mode'])
    writer.writerows(timings)

print('Wrote', out_path)
