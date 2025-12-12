import pytest
import h5py
import numpy as np
from emergent.fish_passage.io import initialize_hecras_geometry


class DummySim:
    def __init__(self, hdfpath):
        self.hdf5 = h5py.File(hdfpath, 'w')


def make_minimal_plan(path):
    with h5py.File(path, 'w') as f:
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)


def test_strict_missing_field_raises(tmp_path):
    plan = tmp_path / 'plan.h5'
    make_minimal_plan(str(plan))
    sim = DummySim(str(tmp_path / 'sim.h5'))

    # Request strict behavior: mapping should raise because required fields are missing
    with pytest.raises(Exception):
        initialize_hecras_geometry(sim, str(plan), create_rasters=True, strict_missing_fields=True)
