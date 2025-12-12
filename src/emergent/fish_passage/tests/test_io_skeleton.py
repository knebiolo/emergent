import h5py
import numpy as np
from emergent.fish_passage.io import initialize_hecras_geometry


class DummySim:
    pass


def make_plan(path):
    with h5py.File(path, 'w') as f:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)


def test_initialize_hecras_geometry_smoke(tmp_path):
    plan = tmp_path / 'plan.h5'
    make_plan(str(plan))
    sim = DummySim()
    sim.hdf5 = h5py.File(str(tmp_path / 'sim_hdf.h5'), 'w')
    try:
        res = initialize_hecras_geometry(sim, str(plan), depth_threshold=0.05, create_rasters=True)
        assert 'coords' in res
        assert res['n_cells'] == 3
        assert 'x_coords' in sim.hdf5
    finally:
        sim.hdf5.close()
