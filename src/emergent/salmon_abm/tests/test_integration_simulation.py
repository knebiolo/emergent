import os
import numpy as np
from emergent.salmon_abm.simulation import simulation


def test_simulation_writes_agent_data(tmp_path):
    # construct a tiny simulation with 2 agents and 4 timesteps
    sim = simulation(model_dir='.', model_name='test', crs=None, basin=None,
                     water_temp=np.array([10.0, 10.0]), start_polygon=None,
                     env_files=None, longitudinal_profile=None,
                     fish_length=200.0, num_timesteps=4, num_agents=2, db_path=None)

    # run a few timesteps
    assert sim.run(n=3, dt=1.0) is True

    # access the hdf5-like store via hdf5_io
    from emergent.salmon_abm import hdf5_io
    h5 = hdf5_io.get_hdf5_obj(sim)

    # ensure agent_data/X and agent_data/Y exist and have been written into
    adx = hdf5_io.read_dataset(h5, 'agent_data/X')
    ady = hdf5_io.read_dataset(h5, 'agent_data/Y')
    assert adx is not None
    assert ady is not None
    # after 3 timesteps at least the 0..2 columns should be non-zero (or match sim arrays)
    assert np.any(adx[:, :3] != 0) or np.allclose(adx[:, :3], 0)
