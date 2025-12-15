import h5py
import numpy as np
from pathlib import Path

from emergent.fish_passage.io import initialize_hdf5, timestep_flush


class DummySim:
    def __init__(self, path: Path, num_agents: int):
        self.hdf5 = h5py.File(str(path), 'w')
        self.num_agents = num_agents


def test_initialize_hdf5_creates_datasets(tmp_path):
    sim_file = tmp_path / 'sim_init.h5'
    sim = DummySim(sim_file, num_agents=3)

    initialize_hdf5(sim, num_agents=3, num_timesteps=10, model_name='testmod')

    assert 'agent_data' in sim.hdf5
    ag = sim.hdf5['agent_data']
    # check a couple of expected datasets
    assert 'X' in ag
    assert ag['X'].shape == (3, 10)
    assert 'sex' in ag
    assert ag['sex'].shape == (3,)

    sim.hdf5.close()


def test_timestep_flush_writes_slice(tmp_path):
    sim_file = tmp_path / 'sim_step.h5'
    sim = DummySim(sim_file, num_agents=2)

    initialize_hdf5(sim, num_agents=2, num_timesteps=5)

    # populate simulation attributes expected by timestep_flush
    sim.X = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],[10.0,11.0,12.0,13.0,14.0]])
    sim.Y = sim.X + 0.5
    sim.battery = np.full((2,5), 0.9)

    # write timestep 2
    timestep_flush(sim, timestep=2, flush_interval=100)

    ag = sim.hdf5['agent_data']
    assert np.allclose(np.asarray(ag['X'][:,2]), np.array([2.0,12.0], dtype='f4'))
    assert np.allclose(np.asarray(ag['Y'][:,2]), np.array([2.5,12.5], dtype='f4'))
    assert np.allclose(np.asarray(ag['battery'][:,2]), np.array([0.9,0.9], dtype='f4'))

    sim.hdf5.close()


def test_enviro_import_with_array(tmp_path):
    from emergent.fish_passage.io import enviro_import

    sim_file = tmp_path / 'sim_env.h5'
    sim = DummySim(sim_file, num_agents=1)

    arr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype='f4')
    enviro_import(sim, arr, surface_type='depth')

    assert 'environment' in sim.hdf5
    env = sim.hdf5['environment']
    assert 'depth' in env
    read = np.asarray(env['depth'])
    assert read.shape == arr.shape
    assert np.allclose(read, arr)

    sim.hdf5.close()


def test_initialize_mental_and_refugia_maps(tmp_path):
    from emergent.fish_passage.io import initialize_mental_map, initialize_refugia_map

    sim_file = tmp_path / 'sim_maps.h5'
    class S:
        pass

    sim = S()
    sim.hdf5 = h5py.File(str(sim_file), 'w')
    sim.height = 40
    sim.width = 60
    sim.num_agents = 3
    sim.depth_rast_transform = None

    initialize_mental_map(sim, avoid_cell_size=10.0)
    assert 'memory' in sim.hdf5
    mem = sim.hdf5['memory']
    assert '0' in mem
    assert mem['0'].shape[0] > 0

    initialize_refugia_map(sim, refugia_cell_size=15.0)
    assert 'refugia' in sim.hdf5
    ref = sim.hdf5['refugia']
    assert '0' in ref
    assert ref['0'].shape[0] > 0

    sim.hdf5.close()
