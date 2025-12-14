import os
import h5py
import numpy as np
import pytest

from emergent.fish_passage.io import ensure_hdf_coords_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan, create_sim_hdf


class FakeSim:
    def __init__(self, hdf):
        self.hdf5 = hdf


def test_writes_coords_at_root(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_root.h5')
    sim_hdf = create_sim_hdf(tmp_path / 'sim_root.h5')
    sim = FakeSim(sim_hdf)

    ensure_hdf_coords_from_hecras(sim, str(plan))

    assert 'x_coords' in sim.hdf5
    assert 'y_coords' in sim.hdf5
    x = sim.hdf5['x_coords'][:]
    y = sim.hdf5['y_coords'][:]
    assert x.size == y.size


def test_rasterize_target_shape(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_raster.h5')
    sim_hdf = create_sim_hdf(tmp_path / 'sim_raster.h5')
    sim = FakeSim(sim_hdf)

    ensure_hdf_coords_from_hecras(sim, str(plan), target_shape=(4, 4))

    assert 'x_coords' in sim.hdf5
    x = sim.hdf5['x_coords'][:]
    assert x.shape == (4, 4)


def test_preserve_existing_coords(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_preserve.h5')
    sim_hdf = create_sim_hdf(tmp_path / 'sim_preserve.h5')
    # pre-create x_coords/y_coords and ensure we don't overwrite
    sim_hdf.create_dataset('x_coords', data=np.ones((2, 2)))
    sim_hdf.create_dataset('y_coords', data=np.ones((2, 2))*2.0)

    sim = FakeSim(sim_hdf)
    ensure_hdf_coords_from_hecras(sim, str(plan))

    # values should remain the same as pre-created
    assert np.allclose(sim.hdf5['x_coords'][:], np.ones((2, 2)))
    assert np.allclose(sim.hdf5['y_coords'][:], np.ones((2, 2))*2.0)
