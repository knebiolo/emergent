"""Start a Nuyakuk-style simulation and stream agent positions to a live viewer.

Usage:
  python tools/start_nuyakuk_live.py --host 127.0.0.1 --port 9001 --nagents 200 --nsteps 100 --fps 20

The script uses data/salmon_abm/start_loc_river_right.shp for seeding agents and
loads environment rasters from data/salmon_abm. It enables the simulation's
`viewer_live` streaming path which sends frames as length-prefixed `.npy` bytes
by default. Use `--raw` to send the raw 'R' protocol instead.
"""
import os
import argparse
import time

from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import io, hdf5_io
import numpy as np


def discover_env_files(base_dir):
    keys = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    out = []
    for fn in keys:
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9001)
    parser.add_argument('--nagents', type=int, default=200)
    parser.add_argument('--nsteps', type=int, default=1000)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--raw', action='store_true', help='Stream using raw R protocol')
    parser.add_argument('--model_name', default='nuyakuk_live')
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
    base = os.path.normpath(base)
    env_files = discover_env_files(base)
    start_poly = os.path.join(base, 'start_loc_river_right.shp')

    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'nuyakuk_live'))
    os.makedirs(outdir, exist_ok=True)

    sim = simulation(
        model_dir=outdir,
        model_name=args.model_name,
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=env_files,
        longitudinal_profile=os.path.join(base, 'longitudinal.shp'),
        num_timesteps=args.nsteps,
        num_agents=args.nagents,
        db_path=os.path.join(outdir, f'{args.model_name}.h5')
    )

    # Import rasters into the simulation HDF5 so sampling returns real values
    try:
        h5 = hdf5_io.get_hdf5_obj(sim)
        for ef in env_files:
            try:
                arr, transform, crs = io.enviro_import(ef)
                key = 'environment/' + os.path.splitext(os.path.basename(ef))[0]
                hdf5_io.write_dataset(h5, key, np.array(arr))
                print('Imported raster into HDF5:', key)
                # attach a simple transform tuple on the sim for depth->geo mapping
                if os.path.basename(ef).startswith('depth'):
                    try:
                        # convert rasterio transform to a 6-tuple (a,b,c,d,e,f)
                        t = transform
                        sim.depth_rast_transform = (t.a, t.b, t.c, t.d, t.e, t.f)
                    except Exception:
                        pass
            except Exception as e:
                print('Failed to import raster', ef, e)

        # write x/y coords if depth was imported
        try:
            depth_ds = hdf5_io.read_dataset(h5, 'environment/depth')
            if depth_ds is not None:
                nrows, ncols = np.array(depth_ds).shape
                a, b, c, d, e, f = getattr(sim, 'depth_rast_transform', (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
                cols = np.arange(ncols, dtype=float)
                rows = np.arange(nrows, dtype=float)
                col_indices, row_indices = np.meshgrid(cols, rows)
                x_coords = a * col_indices + b * row_indices + c
                y_coords = d * col_indices + e * row_indices + f
                hdf5_io.write_dataset(h5, 'environment/x_coords', x_coords)
                hdf5_io.write_dataset(h5, 'environment/y_coords', y_coords)
        except Exception:
            pass

        # sample depth now to ensure values exist
        try:
            depth_vals = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth')
            print('Sampled depth for agents (first 10):', depth_vals[:10])
        except Exception as e:
            print('Depth sampling failed during init:', e)
    except Exception as e:
        print('Raster import into HDF5 failed:', e)

    # Run with viewer_live enabled so the simulation will accept one TCP client
    # and stream frames. The `run` method uses viewer_fps only for pacing hints.
    try:
        status = sim.run(n=args.nsteps, dt=1.0, return_status=True, viewer_live=True,
                         viewer_host=args.host, viewer_port=args.port,
                         viewer_stream_raw=args.raw, viewer_fps=args.fps)
        print('Run finished status:', status)
    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        sim.close()


if __name__ == '__main__':
    main()
