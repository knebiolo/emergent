import tempfile
import os
import h5py
import numpy as np
from emergent.fish_passage.io import HECRASMap


def make_minimal_hecras_hdf(path):
    with h5py.File(path, 'w') as h:
        grp = h.require_group('/Geometry/2D Flow Areas/2D area')
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        grp.create_dataset('Cells Center Coordinate', data=coords)
        # create a simple field aligned to coords
        grp.create_dataset('Cells Minimum Elevation', data=np.array([10.0, 11.0, 12.0]))


def test_hecrasmap_idw(tmp_path):
    p = tmp_path / 'plan.h5'
    make_minimal_hecras_hdf(str(p))
    mapper = HECRASMap(str(p), field_names=['Cells Minimum Elevation'])
    pts = np.array([[0.1, 0.0], [1.5, 0.0]])
    out = mapper.map_idw(pts, k=2)
    assert 'Cells Minimum Elevation' in out
    vals = out['Cells Minimum Elevation']
    assert vals.shape == (2,)
    # assert values are within expected range
    assert 10.0 <= vals[0] <= 11.0
    assert 11.0 <= vals[1] <= 12.0
import tempfile
import h5py
import numpy as np
from emergent.fish_passage.io import initialize_hecras_geometry

class DummySim:
    pass


def make_plan(path):
    # create minimal HECRAS-like HDF5 structure with cell centers and a depth field
    with h5py.File(path, 'w') as f:
        coords = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        # create simple datasets for the fields expected by initialize_hecras_geometry
        base = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/'
        # create one timestep with a value per cell (4 cells)
        vals = np.array([[0.0, 0.1, 0.2, 0.3]])
        f.create_dataset(base + 'Cell Hydraulic Depth', data=vals)
        f.create_dataset(base + 'Cell Velocity - Velocity X', data=np.array([[0.0, 0.0, 0.0, 0.0]]))
        f.create_dataset(base + 'Cell Velocity - Velocity Y', data=np.array([[0.0, 0.0, 0.0, 0.0]]))
        f.create_dataset(base + 'Water Surface', data=np.array([[0.1, 0.1, 0.1, 0.1]]))


def test_initialize_hecras_geometry_smoke(tmp_path):
    plan = tmp_path / 'plan.h5'
    make_plan(str(plan))
    sim = DummySim()
    # create an in-memory hdf5 file for simulation.hdf5
    sim.hdf5 = h5py.File(str(tmp_path / 'sim_hdf.h5'), 'w')
    try:
        res = initialize_hecras_geometry(sim, str(plan), depth_threshold=0.05, create_rasters=True)
        assert 'coords' in res
        assert res['coords'].shape[0] == 4
        # ensure hdf datasets added
        assert 'x_coords' in sim.hdf5
        assert 'y_coords' in sim.hdf5
    finally:
        sim.hdf5.close()
