import os
import h5py
import numpy as np
from emergent.salmon_abm.simulation import simulation

def test_initialize_headings_from_db_smoke():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    start_poly = os.path.join(project_root, 'data', 'salmon_abm', 'start_loc_river_right.shp')
    sim = simulation(model_dir='.', model_name='test', crs=None, basin=None, water_temp=10.0,
                     start_polygon=start_poly, env_files=[], longitudinal_profile=None,
                     num_timesteps=2, num_agents=10, db_path=None)
    # Import available rasters into sim DB so initialization can sample them
    from emergent.salmon_abm import io as abm_io
    rasters = ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']
    data_dir = os.path.join(project_root, 'data', 'salmon_abm')
    for r in rasters:
        p = os.path.join(data_dir, r)
        if os.path.exists(p):
            try:
                abm_io.write_raster_to_hdf5(sim.db, p, dataset_name=os.path.splitext(r)[0], sim=sim)
            except Exception:
                pass
    ok = sim.initialize_headings_from_db()
    assert ok is True
    # headings internal
    assert np.all(np.isfinite(sim.heading)), 'Some headings are not finite'
    # ideal_sog populated
    assert np.all(sim.ideal_sog > 0), 'ideal_sog not populated'
    # HDF5 contains agent_data/ideal_sog
    with h5py.File(sim.db_path, 'r') as f:
        assert 'agent_data/ideal_sog' in f
        arr = np.array(f['agent_data/ideal_sog'])
        assert arr.shape[0] == sim.num_agents
        assert np.all(arr[:,0] > 0)
