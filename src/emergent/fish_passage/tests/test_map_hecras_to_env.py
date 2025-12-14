import h5py
import numpy as np
import pytest

from emergent.fish_passage.io import map_hecras_to_env_rasters, ensure_hdf_coords_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan, create_sim_hdf


class AdapterDict:
    def __init__(self, shape, values):
        self.shape = shape
        self.values = values

    def map_idw(self, pts, k=1):
        # return dict mapping to a flat array
        return {'depth': np.array(self.values)}


class AdapterArray:
    def __init__(self, shape, values):
        self.shape = shape
        self.values = values

    def map_idw(self, pts, k=1):
        return np.array(self.values)


def make_sim_with_env(tmp_path, shape=(2, 2)):
    sim = type('S', (), {})()
    sim.hdf5 = create_sim_hdf(tmp_path / 'sim_env.h5')
    # create environment group and coords
    env = sim.hdf5.create_group('environment')
    h, w = shape
    x = np.arange(h * w).reshape((h, w)).astype('f4')
    y = (np.arange(h * w).reshape((h, w)) * 2).astype('f4')
    sim.hdf5.create_dataset('x_coords', data=x)
    sim.hdf5.create_dataset('y_coords', data=y)
    return sim


def test_map_writes_env_dataset_from_dict_adapter(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_map1.h5')
    sim = make_sim_with_env(tmp_path)
    # adapter returns an array sized 4 (2x2)
    adapter = AdapterDict((2, 2), [0.1, 0.2, 0.3, 0.4])
    sim._hecras_maps = {(str(''), tuple(['depth'])): adapter}

    res = map_hecras_to_env_rasters(sim, plan_path=str(plan), field_names=['depth'], k=1)
    assert res is True
    assert 'environment' in sim.hdf5
    assert 'depth' in sim.hdf5['environment']
    assert sim.hdf5['environment']['depth'].shape == (2, 2)


def test_map_writes_env_dataset_from_array_adapter(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_map2.h5')
    sim = make_sim_with_env(tmp_path)
    adapter = AdapterArray((2, 2), [0.1, 0.2, 0.3, 0.4])
    sim._hecras_maps = {(str(''), tuple(['depth'])): adapter}

    res = map_hecras_to_env_rasters(sim, plan_path=str(plan), field_names=['depth'], k=1)
    assert res is True
    assert sim.hdf5['environment']['depth'].shape == (2, 2)


def test_size_mismatch_results_in_nan_fill(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_map3.h5')
    sim = make_sim_with_env(tmp_path)
    # adapter returns wrong size
    adapter = AdapterArray((2, 2), [0.1, 0.2])
    sim._hecras_maps = {(str(''), tuple(['depth'])): adapter}

    res = map_hecras_to_env_rasters(sim, plan_path=str(plan), field_names=['depth'], k=1)
    assert res is True
    out = sim.hdf5['environment']['depth'][:]
    assert np.all(np.isnan(out))
