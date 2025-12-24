import numpy as np
from emergent.salmon_abm.simulation import simulation

import os
data_dir = r"C:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm"
vel_x_path = os.path.join(data_dir, 'vel_x.tif')
vel_y_path = os.path.join(data_dir, 'vel_y.tif')
vel_mag_path = os.path.join(data_dir, 'vel_mag.tif')
vel_dir_path = os.path.join(data_dir, 'vel_dir.tif')

# Create simulation without env_files, then load rasters into its DB so sampling works
sim = simulation(model_dir='outputs/diagnostics', model_name='test_social', crs=None, basin=None, water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None, num_timesteps=1, num_agents=4)

# load rasters and write into sim.db so simulation.sample_environment reads them
from emergent.salmon_abm import io as io_mod, hdf5_io as h5mod
try:
    # vel_x, vel_y
    if os.path.exists(vel_x_path) and os.path.exists(vel_y_path):
        vx_arr, vx_tr, vx_crs = io_mod.enviro_import(vel_x_path)
        vy_arr, vy_tr, vy_crs = io_mod.enviro_import(vel_y_path)
        h5mod.write_dataset(sim.db, 'environment/vel_x', vx_arr)
        h5mod.write_dataset(sim.db, 'environment/vel_y', vy_arr)
        # set raster transforms on sim (convert Affine to tuple)
        try:
            sim.depth_rast_transform = tuple(vx_tr)
            sim.vel_dir_rast_transform = tuple(vx_tr)
            sim.vel_mag_rast_transform = tuple(vx_tr)
        except Exception:
            pass
    # vel_mag + vel_dir fallback
    if os.path.exists(vel_mag_path) and os.path.exists(vel_dir_path):
        mag_arr, mag_tr, mag_crs = io_mod.enviro_import(vel_mag_path)
        dir_arr, dir_tr, dir_crs = io_mod.enviro_import(vel_dir_path)
        h5mod.write_dataset(sim.db, 'environment/vel_mag', mag_arr)
        h5mod.write_dataset(sim.db, 'environment/vel_dir', dir_arr)
        try:
            sim.depth_rast_transform = tuple(mag_tr)
            sim.vel_dir_rast_transform = tuple(mag_tr)
            sim.vel_mag_rast_transform = tuple(mag_tr)
        except Exception:
            pass
except Exception as e:
    print('Warning: could not load env rasters into sim DB:', e)
sim.current_step = 0
# Use the simulation API to (re)initialize headings from the DB after rasters were written
try:
    sim.initialize_headings_from_db()
except Exception:
    pass

# Enable detailed behavior debugging so alignment_cue prints and NPZ includes alignment_diag
sim.debug_behavior = True
sim.debug = True

# Ensure PID tuning disabled so alignment and cohesion cues run
try:
    sim.pid_tuning = False
    print('sim.pid_tuning set to', sim.pid_tuning)
except Exception:
    pass

# Diagnostic prints: report whether headings were set and what values were sampled
try:
    from emergent.salmon_abm import hdf5_io as _h5, utils as _utils
    h5obj = _h5.get_hdf5_obj(sim)
    ok = sim.initialize_headings_from_db()
    print('initialize_headings_from_db returned', ok)
    # sample components at agent positions for inspection
    vx_ds = _h5.read_dataset(h5obj, 'environment/vel_x', default=None)
    vy_ds = _h5.read_dataset(h5obj, 'environment/vel_y', default=None)
    if vx_ds is not None and vy_ds is not None:
        vx_arr = np.array(vx_ds)
        vy_arr = np.array(vy_ds)
        rows, cols = _utils.geo_to_pixel(sim.X, sim.Y, sim.depth_rast_transform)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        valid = (rows >= 0) & (cols >= 0) & (rows < vx_arr.shape[0]) & (cols < vx_arr.shape[1])
        sampled_vx = np.full(sim.num_agents, np.nan)
        sampled_vy = np.full(sim.num_agents, np.nan)
        if np.any(valid):
            sampled_vx[valid] = vx_arr[rows[valid], cols[valid]]
            sampled_vy[valid] = vy_arr[rows[valid], cols[valid]]
        print('sampled vx:', sampled_vx)
        print('sampled vy:', sampled_vy)
        print('valid indices:', valid)
    else:
        print('vel_x/vel_y datasets not found in sim.db')
except Exception as e:
    print('diagnostic prints failed:', e)

# Always print sim.heading to confirm initialization
try:
    print('sim.heading after initialize:', sim.heading)
    print('sim.x_vel (pre-timestep):', getattr(sim, 'x_vel', None))
    print('sim.y_vel (pre-timestep):', getattr(sim, 'y_vel', None))
except Exception as e:
    print('failed to print sim.heading/x_vel/y_vel:', e)
except Exception as e:
    print('diagnostic prints failed:', e)

sim.timestep(0, 1.0)

# Print alignment diagnostics stored on sim by behavior (if present)
try:
    print('sim._alignment_diag present:', hasattr(sim, '_alignment_diag'))
    if hasattr(sim, '_alignment_diag'):
        print('alignment_diag keys:', list(sim._alignment_diag.keys()))
        for k, v in sim._alignment_diag.items():
            print(k, type(v), getattr(v, 'shape', v))
except Exception as e:
    print('failed to print sim._alignment_diag:', e)

# Also print sim.heading and sampled velocities after timestep to verify alignment use
try:
    print('sim.heading after timestep:', sim.heading)
    print('sim.x_vel (post-timestep):', getattr(sim, 'x_vel', None))
    print('sim.y_vel (post-timestep):', getattr(sim, 'y_vel', None))
    if hasattr(sim, '_alignment_diag'):
        ad = sim._alignment_diag
        try:
            print('\nAlignment diagnostics sample:')
            for k in ('raw_headings_neighbors', 'headings_neighbors_used', 'used_velocity_heading', 'neighbor_indices', 'agent_indices'):
                if k in ad:
                    v = ad[k]
                    try:
                        print(k, type(v), getattr(v, 'shape', None), '\n sample=', v[:20])
                    except Exception:
                        print(k, type(v), getattr(v, 'shape', None))
        except Exception:
            pass
except Exception:
    pass

import glob, os
files = glob.glob('outputs/diagnostics/behavior_debug_step_*.npz')
if not files:
    raise SystemExit('No behavior NPZs found')
f = max(files, key=os.path.getmtime)
data = np.load(f)
print('Inspecting', f)
print('keys:', list(data.keys()))
for k in ['avoid_mag','collision_mag','alignment_mag','cohesion_mag','neighbor_counts','neighbor_mean_distance','neighbor_any_within_2bl','neighbor_headings','neighbor_rel_heading','closest_agent','nearest_neighbor_distance']:
    if k in data:
        try:
            print(k, data[k])
        except Exception:
            print(k, 'could not print array')
    else:
        print(k, 'missing')

# basic assertions
assert np.any(data['cohesion_mag'] > 0), 'cohesion_mag not > 0'
assert np.any(np.isfinite(data['alignment_mag'])), 'alignment_mag not finite'
print('Social cues test passed')
