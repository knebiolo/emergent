import os
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import numpy as np

base = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(base, 'data', 'salmon_abm')
start_poly = os.path.join(data_dir, 'start_loc_river_right.shp')
env_files = [os.path.join(data_dir, fname) for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']]

outdir = os.path.abspath(os.path.join(base, 'outputs', 'diagnostics_start1000'))
os.makedirs(outdir, exist_ok=True)

print('Starting sim with start polygon:', start_poly)
sim = simulation(model_dir=outdir, model_name='start1000', crs=None, basin='Nushagak River', water_temp=10.0, start_polygon=start_poly, env_files=env_files, longitudinal_profile=None, num_timesteps=1, num_agents=1000)
print('num_agents:', sim.num_agents)

# Import env rasters into the simulation DB so heading initialization can sample them
try:
    h5obj = getattr(sim, 'db', None)
    if h5obj is not None:
        for ef in env_files:
            try:
                arr, transform, crs = io.enviro_import(ef)
                key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
                hdf5_io.write_dataset(h5obj, key, arr)
                # keep last transform for assigning sim transforms
                last_transform = transform
            except Exception as e:
                print('env import failed for', ef, e)

        # write x/y coords if we have a raster shape
        try:
            depth_ds = hdf5_io.read_dataset(h5obj, 'environment/depth')
            if depth_ds is not None:
                nrows, ncols = depth_ds.shape
                # prefer returned affine transform
                affine = None
                try:
                    if last_transform is not None:
                        affine = (last_transform.a, last_transform.b, last_transform.c, last_transform.d, last_transform.e, last_transform.f)
                except Exception:
                    try:
                        affine = tuple(last_transform)
                    except Exception:
                        affine = None
                if affine is not None:
                    a, b, c, d, e, f = affine
                    cols = np.arange(ncols, dtype=float)
                    rows_idx = np.arange(nrows, dtype=float)
                    col_indices, row_indices = np.meshgrid(cols, rows_idx)
                    x_coords = a * col_indices + b * row_indices + c
                    y_coords = d * col_indices + e * row_indices + f
                    sim.depth_rast_transform = (a, b, c, d, e, f)
                    sim.vel_mag_rast_transform = (a, b, c, d, e, f)
                    sim.vel_dir_rast_transform = (a, b, c, d, e, f)
                    sim.refugia_map_transform = (a, b, c, d, e, f)
                else:
                    x_coords = np.tile(np.arange(ncols, dtype=float), (nrows, 1))
                    y_coords = np.tile(np.arange(nrows, dtype=float)[:, np.newaxis], (1, ncols))
                    sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                    sim.vel_mag_rast_transform = sim.depth_rast_transform
                    sim.vel_dir_rast_transform = sim.depth_rast_transform
                    sim.refugia_map_transform = sim.depth_rast_transform

                hdf5_io.write_dataset(h5obj, 'environment/x_coords', x_coords)
                hdf5_io.write_dataset(h5obj, 'environment/y_coords', y_coords)
        except Exception:
            pass

        # Now (re)initialize headings from the DB so they reflect environment rasters
        try:
            ok = sim.initialize_headings_from_db()
            print('initialize_headings_from_db returned', ok)
        except Exception as e:
            print('initialize_headings_from_db failed:', e)
except Exception:
    # tolerate any failures in env import / init steps and proceed to report positions
    pass
print('X stats: min/max/mean:', float(np.min(sim.X)), float(np.max(sim.X)), float(np.mean(sim.X)))
print('Y stats: min/max/mean:', float(np.min(sim.Y)), float(np.max(sim.Y)), float(np.mean(sim.Y)))
print('Sample positions (first 10):')
for i in range(min(10, sim.num_agents)):
    print(i, sim.X[i], sim.Y[i])

# pixel mapping
from emergent.salmon_abm import utils
rows, cols = utils.geo_to_pixel(sim.X, sim.Y, sim.depth_rast_transform)
unique_pixels = set(zip(rows.tolist(), cols.tolist()))
print('Unique pixel cells occupied by agents:', len(unique_pixels))
print('Example pixel cells (first 10):', list(unique_pixels)[:10])
print('Heading stats (min/max/mean):', float(np.min(sim.heading)), float(np.max(sim.heading)), float(np.mean(sim.heading)))

# run only one timestep to ensure movement/velocity sampling occurs
sim.timestep(0, 1.0)
print('After one timestep: mean x_vel, y_vel:', float(np.mean(sim.x_vel)), float(np.mean(sim.y_vel)))
