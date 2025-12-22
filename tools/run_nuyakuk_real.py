"""Runner: initialize a real model using data/salmon_abm inputs and run a short smoke test.

This script is intended to be run from the repository root. It will:
- map rasters from data/salmon_abm to expected env keys
- import longitudinal and start polygon shapefiles
- instantiate `simulation` with start_loc_river_right.shp and run 10 steps
"""
import os
from emergent.salmon_abm import io, hdf5_io
from emergent.salmon_abm.simulation import simulation


def discover_env_files(base_dir):
    keys = {
        'vel_x': 'vel_x.tif',
        'vel_y': 'vel_y.tif',
        'vel_dir': 'vel_dir.tif',
        'vel_mag': 'vel_mag.tif',
        'depth': 'depth.tif',
        'elev': 'elev.tif',
        'wsel': 'wsel.tif',
        'wetted': 'wetted_perimeter.tif',
    }
    out = {}
    for k, fn in keys.items():
        p = os.path.join(base_dir, fn)
        if os.path.exists(p):
            out[k] = p
    return out


def main():
    base = os.path.join('data', 'salmon_abm')
    env_files = discover_env_files(base)
    print('Discovered env files:', env_files)

    longitudinal = os.path.join(base, 'longitudinal.shp')
    start_poly = os.path.join(base, 'start_loc_river_right.shp')

    model_dir = os.path.join('outputs', 'nuyakuk_real')
    os.makedirs(model_dir, exist_ok=True)

    sim = simulation(
        model_dir=model_dir,
        model_name='nuyakuk_real',
        crs=None,
        basin='nuyakuk',
        water_temp=10.0,
        start_polygon=start_poly,
        env_files=list(env_files.values()),
        longitudinal_profile=longitudinal,
        num_timesteps=100,
        num_agents=50,
        db_path=os.path.join(model_dir, 'nuyakuk_real.h5')
    )

    # import rasters into HDF5 using io helpers where possible
    try:
        h = sim.db
        # load rasters and write to environment group
        for key, path in env_files.items():
            try:
                arr, tr, crs = io.enviro_import(path)
                hdf5_io.write_dataset(h, f'environment/{key}', arr)
                print('Imported', key)
            except Exception as e:
                print('Failed to import', key, e)
    except Exception as e:
        print('Could not write env files to HDF5:', e)

    # import longitudinal profile
    try:
        gdf = io.longitudinal_import(longitudinal)
        # store minimal vector info
        hdf5_io.write_dataset(sim.db, 'environment/longitudinal_count', len(gdf))
        print('Imported longitudinal shape, features:', len(gdf))
    except Exception as e:
        print('Longitudinal import failed:', e)

    # run a short smoke test
    status = sim.run(n=10, dt=1.0, return_status=True)
    print('Run status:', status)
    sim.close()


if __name__ == '__main__':
    main()
