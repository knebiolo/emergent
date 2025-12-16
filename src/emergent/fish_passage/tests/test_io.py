import os
import tempfile

from emergent.fish_passage.io import safe_flush


def test_safe_flush_h5py_file(tmp_path):
    try:
        import h5py
    except Exception:
        # If h5py not installed, skip this test by asserting no-op behavior
        safe_flush(object())
        return

    fp = tmp_path / 'test.h5'
    with h5py.File(str(fp), 'w') as f:
        f.create_dataset('a', data=[1, 2, 3])
        # calling safe_flush should not raise
        safe_flush(f)

    # reopen and ensure dataset exists
    with h5py.File(str(fp), 'r') as f:
        assert 'a' in f


def test_safe_flush_dummy_object():
    class NoFlush:
        pass

    # Should not raise
    safe_flush(NoFlush())
import h5py
import numpy as np

from emergent.fish_passage import io as fp_io


class DummySim:
    def __init__(self, h5file, num_agents, num_timesteps):
        self.hdf5 = h5file
        self.num_agents = int(num_agents)
        self.num_timesteps = int(num_timesteps)
        # attributes that timestep_flush may read
        self.X = np.zeros((self.num_agents, self.num_timesteps), dtype=float)
        self.Y = np.zeros((self.num_agents, self.num_timesteps), dtype=float)
        self.battery = np.zeros((self.num_agents, self.num_timesteps), dtype=float)


def test_initialize_hdf5_creates_expected_datasets(tmp_path):
    path = tmp_path / "test_io.h5"
    num_agents = 4
    num_timesteps = 10
    with h5py.File(path, 'w') as h5:
        sim = DummySim(h5, num_agents, num_timesteps)
        # call initializer
        fp_io.initialize_hdf5(sim, num_agents, num_timesteps, model_name='unittest')

        # agent_data group exists
        assert 'agent_data' in h5
        ag = h5['agent_data']

        # check a representative set of datasets and shapes
        expected_2d = ['X', 'Y', 'prev_X', 'prev_Y', 'heading', 'sog', 'swim_speed', 'battery', 'drag', 'thrust']
        for name in expected_2d:
            assert name in ag, f"Missing dataset {name}"
            ds = ag[name]
            assert ds.shape == (num_agents, num_timesteps)

        expected_1d = ['sex', 'length', 'ucrit', 'weight', 'body_depth']
        for name in expected_1d:
            assert name in ag, f"Missing dataset {name}"
            ds = ag[name]
            assert ds.shape == (num_agents,)

        # check file attributes
        assert h5.attrs.get('simulation_name', '').startswith('unittest')
        assert int(h5.attrs.get('num_agents', -1)) == num_agents
        assert int(h5.attrs.get('num_timesteps', -1)) == num_timesteps


def test_timestep_flush_writes_column(tmp_path):
    path = tmp_path / "test_io_flush.h5"
    num_agents = 3
    num_timesteps = 5
    with h5py.File(path, 'w') as h5:
        sim = DummySim(h5, num_agents, num_timesteps)
        # initialize datasets
        fp_io.initialize_hdf5(sim, num_agents, num_timesteps, model_name='flush_test')

        # set some values in sim arrays for timestep 0
        sim.X[:, 0] = np.array([1.1, 2.2, 3.3])
        sim.Y[:, 0] = np.array([4.4, 5.5, 6.6])
        sim.battery[:, 0] = np.array([0.1, 0.2, 0.3])

        # call flush for timestep 0 (use flush_interval=1 to force flush)
        fp_io.timestep_flush(sim, timestep=0, flush_interval=1)

        ag = h5['agent_data']
        # verify the first column of X/Y/battery were written
        np.testing.assert_allclose(ag['X'][:, 0], sim.X[:, 0].astype('float32'))
        np.testing.assert_allclose(ag['Y'][:, 0], sim.Y[:, 0].astype('float32'))
        np.testing.assert_allclose(ag['battery'][:, 0], sim.battery[:, 0].astype('float32'))
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
