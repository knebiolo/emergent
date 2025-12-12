import tempfile
import os
import h5py
import numpy as np
from emergent.fish_passage.io import ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters

import h5py
import numpy as np
import pytest

from emergent.fish_passage.io import ensure_hdf_coords_from_hecras, map_hecras_to_env_rasters
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan, create_sim_hdf


class FakeAdapter:
    def __init__(self, grid_shape, values):
        self.grid_shape = grid_shape
        self.values = values

    def map_idw(self, agent_xy, k=1):
        return np.array(self.values)


class FakeSim:
    def __init__(self):
        self.hdf5 = None
        self._hecras_maps = {}
        self.hecras_fields = ['depth']


def test_ensure_hdf_coords_and_map(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan.h5')

    sim = FakeSim()
    sim.hdf5 = create_sim_hdf(tmp_path / 'sim.h5')

    ensure_hdf_coords_from_hecras(sim, str(plan))
    assert 'x_coords' in sim.hdf5
    assert 'y_coords' in sim.hdf5

    grid_shape = (2, 2)
    n = grid_shape[0] * grid_shape[1]
    adapter = FakeAdapter(grid_shape, np.zeros((n,)))
    key = (str(''), tuple(sim.hecras_fields))
    sim._hecras_maps[key] = adapter

    res = map_hecras_to_env_rasters(sim, plan_path=str(plan), field_names=['depth'], k=1)
    assert res is True
    assert 'environment' in sim.hdf5
    assert 'depth' in sim.hdf5['environment']


def test_map_raises_when_no_adapter(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan2.h5')
    sim = FakeSim()
    sim.hdf5 = create_sim_hdf(tmp_path / 'sim2.h5')

    ensure_hdf_coords_from_hecras(sim, str(plan))

    with pytest.raises(Exception):
        map_hecras_to_env_rasters(sim, plan_path=str(plan), field_names=['depth'], k=1)
