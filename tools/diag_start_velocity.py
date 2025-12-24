import os
from emergent.salmon_abm.simulation import simulation

# small diagnostic: 5 agents, 1 timestep
model_dir = 'outputs'
model_name = 'diag_start_velocity'
crs = None
basin = 'Nushagak River'
water_temp = 10.0
start_polygon = 'data/salmon_abm/start_loc_river_right.shp'
# include raster env files if present
env_dir = os.path.join('data', 'salmon_abm')
env_files = []
for fn in ('vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif'):
    p = os.path.join(env_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

print('Using env_files:', env_files)

sim = simulation(model_dir, model_name, crs, basin, water_temp, start_polygon, env_files, longitudinal_profile=None, num_timesteps=2, num_agents=5)

# write env rasters into sim.db so initialize_headings_from_db can sample them
h5obj = getattr(sim, 'db', None)
if h5obj is not None:
    from emergent.salmon_abm import io, hdf5_io
    for ef in env_files:
        try:
            arr, transform, crs = io.enviro_import(ef)
            key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
            hdf5_io.write_dataset(h5obj, key, arr)
            # preserve transform if available
            try:
                if transform is not None and hasattr(transform, 'a'):
                    sim.depth_rast_transform = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
                    sim.vel_mag_rast_transform = sim.depth_rast_transform
                    sim.vel_dir_rast_transform = sim.depth_rast_transform
            except Exception:
                pass
        except Exception as e:
            print('env import->db failed for', ef, e)

# enable debugging to capture movement/frequency internals
sim.debug_freq = True
sim.debug_movement = True

print('\nInitial state:')
print('X[:5]=', sim.X[:5])
print('Y[:5]=', sim.Y[:5])
print('heading[:5]=', sim.heading[:5])
print('sog[:5]=', sim.sog[:5])
print('ideal_sog[:5]=', sim.ideal_sog[:5])
print('x_vel[:5]=', sim.x_vel[:5])
print('y_vel[:5]=', sim.y_vel[:5])
# also print movement-related arrays initial
def maybe_slice(arr):
    try:
        return None if arr is None else arr[:5]
    except Exception:
        return None

print('is_stuck[:5]=', maybe_slice(getattr(sim, 'is_stuck', None)))
print('swim_behav[:5]=', maybe_slice(getattr(sim, 'swim_behav', None)))
print('wave_drag[:5]=', maybe_slice(getattr(sim, 'wave_drag', None)))
print('weight[:5]=', maybe_slice(getattr(sim, 'weight', None)))

# run one timestep
ok = sim.timestep(0, 1.0)
print('\nAfter one timestep:')
print('X[:5]=', sim.X[:5])
print('Y[:5]=', sim.Y[:5])
print('x_vel[:5]=', sim.x_vel[:5])
print('y_vel[:5]=', sim.y_vel[:5])

# print movement internals
print('Hz[:5]=', getattr(sim, 'Hz', None)[:5])
print('prev_Hz[:5]=', getattr(sim, 'prev_Hz', None)[:5])
print('thrust[:5]=', getattr(sim, 'thrust', None)[:5])
print('drag[:5]=', getattr(sim, 'drag', None)[:5])
print('error[:5]=', getattr(sim, 'error', None)[:5])
print('pid_adjustment[:5]=', getattr(sim, 'pid_adjustment', None)[:5])
print('integral[:5]=', getattr(sim, 'integral', None)[:5])

# Also print displacement magnitudes
import numpy as np
disp = np.sqrt((sim.X - sim.prev_X)**2 + (sim.Y - sim.prev_Y)**2)
print('displacements[:5]=', disp[:5])

# print whether any moved
print('any moved?', np.any(disp > 0))

# cleanup
sim.close()
