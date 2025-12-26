"""Run sampling checks across all start polygons and report any nodata/sentinel samples.

Usage: python tools/smoke_validate_polygons.py
"""
import os
import glob
import numpy as np
from emergent.salmon_abm import simulation, io


def check_polygon(shp_path, nagents=20):
    print('\nChecking', shp_path)
    sim = simulation.simulation(model_dir='.', model_name='smoke', crs=None, basin=None, water_temp=10.0,
                                start_polygon=shp_path, env_files=[], longitudinal_profile=None,
                                num_timesteps=2, num_agents=nagents, db_path=None)
    # import rasters
    data_dir = os.path.join(os.path.dirname(shp_path), '.')
    rasters = ['vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    for r in rasters:
        p = os.path.join(os.path.dirname(shp_path), r)
        if os.path.exists(p):
            io.write_raster_to_hdf5(sim.db, p, dataset_name=os.path.splitext(r)[0], sim=sim)

    vx = sim.sample_environment(getattr(sim, 'vel_x_rast_transform', getattr(sim, 'vel_x_rast_transform_tuple', sim.depth_rast_transform)), 'vel_x')
    vy = sim.sample_environment(getattr(sim, 'vel_y_rast_transform', getattr(sim, 'vel_y_rast_transform_tuple', sim.depth_rast_transform)), 'vel_y')

    # sentinel nodata often <= -9000; treat as invalid
    invalid_vx = np.where((~np.isfinite(vx)) | (vx <= -9000), 1, 0)
    invalid_vy = np.where((~np.isfinite(vy)) | (vy <= -9000), 1, 0)
    total_invalid = int(np.sum(invalid_vx) + np.sum(invalid_vy))
    print('invalid vx count:', int(np.sum(invalid_vx)), 'invalid vy count:', int(np.sum(invalid_vy)))
    if total_invalid > 0:
        print('FAIL: sentinel or NaN samples found for', shp_path)
        return False
    print('OK: no sentinel/NaN samples')
    return True


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm')
    patterns = ['start_loc_*shp']
    shp_glob = os.path.join(base, 'start_loc_*.shp')
    files = glob.glob(shp_glob)
    if not files:
        print('No start polygons found under', base)
        return 2
    all_ok = True
    for f in files:
        ok = check_polygon(f)
        all_ok = all_ok and ok
    if not all_ok:
        print('\nOne or more polygons failed sampling checks')
        return 1
    print('\nAll polygons passed sampling checks')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
