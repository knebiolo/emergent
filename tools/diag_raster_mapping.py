"""Diagnostic: sample environment rasters at start polygon agent positions.

Usage: python tools/diag_raster_mapping.py [start_shapefile]
"""
import os
import sys
import numpy as np
from emergent.salmon_abm import simulation, io


def main(start_shp=None, nagents=20):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if start_shp is None:
        start_shp = os.path.join(root, 'data', 'salmon_abm', 'start_loc_river_right.shp')
    print('Start polygon:', start_shp)
    sim = simulation.simulation(model_dir='.', model_name='diag', crs=None, basin=None, water_temp=10.0,
                                start_polygon=start_shp, env_files=[], longitudinal_profile=None,
                                num_timesteps=2, num_agents=nagents, db_path=None)

    # import rasters in data/salmon_abm if present
    data_dir = os.path.join(root, 'data', 'salmon_abm')
    rasters = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    for r in rasters:
        p = os.path.join(data_dir, r)
        if os.path.exists(p):
            try:
                arr, tr, crs = io.write_raster_to_hdf5(sim.db, p, dataset_name=os.path.splitext(r)[0], sim=sim)
                print('Imported', r, 'shape=', getattr(arr, 'shape', None), 'transform=', tr)
            except Exception as e:
                print('Failed to import', r, e)
        else:
            print('Missing raster', r)

    # sample via simulation.sample_environment
    samples = {}
    for key in ('vel_x', 'vel_y', 'vel_mag', 'vel_dir'):
        out = sim.sample_environment(getattr(sim, f'{key}_rast_transform', sim.depth_rast_transform), key)
        samples[key] = out
        nan_count = int(np.sum(~np.isfinite(out)))
        print(f'{key}: nan_count={nan_count}/{len(out)} finite_min={np.nanmin(out):.6g} finite_max={np.nanmax(out):.6g}')
        print('  sample[:5]=', out[:5])

    # report per-agent combined status
    for i in range(sim.num_agents):
        vx = samples['vel_x'][i]
        vy = samples['vel_y'][i]
        vm = samples['vel_mag'][i]
        vd = samples['vel_dir'][i]
        print(f'agent {i}: vx={vx:.6g} vy={vy:.6g} mag={vm:.6g} dir={vd:.6g}')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(start_shp=arg)
