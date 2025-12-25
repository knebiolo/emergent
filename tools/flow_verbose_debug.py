import os
import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io
from emergent.salmon_abm.utils import geo_to_pixel

# Diagnostic: verbose sampling for 10 agents
model_dir = 'outputs'
model_name = 'flow_verbose'
crs = None
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
env_dir = os.path.join('data', 'salmon_abm')
env_files = []
for fn in ('vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif'):
    p = os.path.join(env_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=2, num_agents=10)

h5 = getattr(sim, 'db', None)
if h5 is None:
    print('No DB; abort')
    sim.close()
    raise SystemExit(1)

# Import rasters into HDF5 (if not already) and reinit headings
from emergent.salmon_abm import io as _io
for ef in env_files:
    try:
        arr, transform, crs = _io.enviro_import(ef)
        key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
        hdf5_io.write_dataset(h5, key, arr)
        # set simulation raster transform attributes so geo_to_pixel uses correct affine
        base = os.path.splitext(os.path.basename(ef))[0]
        try:
            if base == 'depth':
                sim.depth_rast_transform = transform
            elif base == 'vel_x':
                sim.vel_x_rast_transform = transform
            elif base == 'vel_y':
                sim.vel_y_rast_transform = transform
            elif base == 'vel_mag':
                sim.vel_mag_rast_transform = transform
            elif base == 'vel_dir':
                sim.vel_dir_rast_transform = transform
            elif base == 'wetted':
                sim.wetted_transform = transform
        except Exception:
            pass
    except Exception as e:
        print('env import failed', ef, e)
try:
    sim.initialize_headings_from_db()
except Exception as e:
    print('initialize_headings_from_db failed', e)

# Read arrays
vx_arr = hdf5_io.read_dataset(h5, 'environment/vel_x')
vy_arr = hdf5_io.read_dataset(h5, 'environment/vel_y')
mag_arr = hdf5_io.read_dataset(h5, 'environment/vel_mag')
dir_arr = hdf5_io.read_dataset(h5, 'environment/vel_dir')

# sample per-agent using velocity raster transform (vel_x/vel_y)
rows, cols = geo_to_pixel(sim.X, sim.Y, getattr(sim, 'vel_x_rast_transform', sim.depth_rast_transform))
rows = np.asarray(rows, dtype=int)
cols = np.asarray(cols, dtype=int)

print('agent | X, Y | sample_row, col | vel_x, vel_y, vel_mag, vel_dir | sim.x_vel, sim.y_vel | heading | init_fish_vel | disp | cos(-water,disp)')

disp_x = None

# run one timestep
sim.timestep(0, 1.0)
disp_x = sim.X - sim.prev_X
disp_y = sim.Y - sim.prev_Y

for i in range(sim.num_agents):
    rx = rows[i]
    cx = cols[i]
    sample_vx = sample_vy = sample_mag = sample_dir = 0.0
    inbounds = True
    try:
        if vx_arr is None or vy_arr is None:
            inbounds = False
        else:
            if rx < 0 or cx < 0 or rx >= np.array(vx_arr).shape[0] or cx >= np.array(vx_arr).shape[1]:
                inbounds = False
            else:
                sample_vx = float(np.array(vx_arr)[rx, cx])
                sample_vy = float(np.array(vy_arr)[rx, cx])
        if mag_arr is not None and dir_arr is not None and inbounds:
            sample_mag = float(np.array(mag_arr)[rx, cx])
            sample_dir = float(np.array(dir_arr)[rx, cx])
    except Exception:
        inbounds = False

    sim_vx = float(getattr(sim, 'x_vel', np.zeros(sim.num_agents))[i])
    sim_vy = float(getattr(sim, 'y_vel', np.zeros(sim.num_agents))[i])
    heading = float(sim.heading[i])
    init_fv = sim.initial_fish_vel[i] if hasattr(sim, 'initial_fish_vel') else (0.0, 0.0)
    dx = float(disp_x[i])
    dy = float(disp_y[i])
    # compute cos with -water
    w = np.array([-sample_vx, -sample_vy])
    d = np.array([dx, dy])
    if np.linalg.norm(w) == 0 or np.linalg.norm(d) == 0:
        cos = np.nan
    else:
        cos = float(np.dot(w, d) / (np.linalg.norm(w) * np.linalg.norm(d)))

    print(f'{i:3d} | {sim.X[i]:8.3f},{sim.Y[i]:8.3f} | {rx:4d},{cx:4d} | {sample_vx:6.3f},{sample_vy:6.3f},{sample_mag:6.3f},{sample_dir:6.3f} | {sim_vx:6.3f},{sim_vy:6.3f} | {heading:6.3f} | {init_fv[0]:6.3f},{init_fv[1]:6.3f} | {dx:6.3f},{dy:6.3f} | {cos:6.3f}')

sim.close()
