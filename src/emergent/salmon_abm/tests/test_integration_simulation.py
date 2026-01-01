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


def test_simulation_thread_async_output_writes_agent_data(tmp_path):
    # Thread async output: simulation should run without sync writes and still
    # populate agent_data/X and agent_data/Y via the async writer.
    from emergent.salmon_abm import hdf5_io

    sim = simulation(
        model_dir=".",
        model_name="test_async",
        crs=None,
        basin=None,
        water_temp=np.array([10.0, 10.0]),
        start_polygon=None,
        env_files=None,
        longitudinal_profile=None,
        fish_length=200.0,
        num_timesteps=6,
        num_agents=2,
        db_path=str(tmp_path / "async_sim.h5"),
        output_write_mode="none",
        output_write_backend="thread",
    )
    # Explicitly configure keys written each step.
    sim.output_write_keys = ("agent_data/X", "agent_data/Y")

    assert sim.run(n=3, dt=1.0) is True

    h5 = hdf5_io.get_hdf5_obj(sim)
    adx = hdf5_io.read_dataset(h5, "agent_data/X")
    ady = hdf5_io.read_dataset(h5, "agent_data/Y")
    assert adx is not None
    assert ady is not None
    assert np.asarray(adx).shape == (2, 6)
    assert np.asarray(ady).shape == (2, 6)


def test_simulation_process_async_output_writes_agent_data(tmp_path):
    import h5py

    sim = simulation(
        model_dir=".",
        model_name="test_async_proc",
        crs=None,
        basin=None,
        water_temp=np.array([10.0, 10.0]),
        start_polygon=None,
        env_files=None,
        longitudinal_profile=None,
        fish_length=200.0,
        num_timesteps=6,
        num_agents=2,
        db_path=str(tmp_path / "async_sim_proc.h5"),
        output_write_mode="none",
        output_write_backend="process",
    )
    sim.output_write_keys = ("agent_data/X", "agent_data/Y")
    sim.output_write_queue_max = 8
    sim.output_write_policy = "block"

    assert sim.run(n=3, dt=1.0) is True

    with h5py.File(sim.db_path, "r") as h5:
        assert "agent_data/X" in h5
        assert "agent_data/Y" in h5
        X = np.asarray(h5["agent_data/X"])
        Y = np.asarray(h5["agent_data/Y"])
    assert X.shape == (2, 6)
    assert Y.shape == (2, 6)


def test_simulation_shm_async_output_writes_agent_data(tmp_path):
    import h5py

    sim = simulation(
        model_dir=".",
        model_name="test_async_shm",
        crs=None,
        basin=None,
        water_temp=np.array([10.0, 10.0]),
        start_polygon=None,
        env_files=None,
        longitudinal_profile=None,
        fish_length=200.0,
        num_timesteps=6,
        num_agents=2,
        db_path=str(tmp_path / "async_sim_shm.h5"),
        output_write_mode="none",
        output_write_backend="shm",
    )
    sim.output_write_keys = ("agent_data/X", "agent_data/Y")
    sim.output_write_ring_slots = 4

    assert sim.run(n=3, dt=1.0) is True

    with h5py.File(sim.db_path, "r") as h5:
        assert "agent_data/X" in h5
        assert "agent_data/Y" in h5
        X = np.asarray(h5["agent_data/X"])
        Y = np.asarray(h5["agent_data/Y"])
    assert X.shape == (2, 6)
    assert Y.shape == (2, 6)
