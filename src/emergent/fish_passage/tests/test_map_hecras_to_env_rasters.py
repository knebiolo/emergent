import tempfile
import h5py
import numpy as np
from pathlib import Path

from emergent.fish_passage.io import HECRASMap, ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


class DummySim:
    def __init__(self, hdf_path):
        self.hdf5 = h5py.File(hdf_path, 'w')
        self._hecras_maps = {}


def test_map_hecras_to_env_rasters_basic(tmp_path: Path):
    plan_file = tmp_path / 'plan.h5'
    create_minimal_plan(plan_file)

    sim_file = tmp_path / 'sim.h5'
    sim = DummySim(sim_file)

    # ensure coords are populated
    ensure_hdf_coords_from_hecras(sim, str(plan_file))

    # register adapter under both exact plan key and fallback empty-plan key
    m = HECRASMap(str(plan_file), field_names=['Fields/depth'])
    sim._hecras_maps = {
        (str(plan_file), tuple(['Fields/depth'])): m,
        ('', tuple(['Fields/depth'])): m,
    }

    # run mapping
    ok = map_hecras_to_env_rasters(sim, str(plan_file), ['Fields/depth'], k=1)
    assert ok is True
    assert 'environment' in sim.hdf5
    assert 'Fields/depth' in sim.hdf5['environment']
    arr = np.asarray(sim.hdf5['environment']['Fields/depth'])
    assert arr.size > 0
    sim.hdf5.close()
