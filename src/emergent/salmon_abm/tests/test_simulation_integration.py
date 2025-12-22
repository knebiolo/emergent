import os
import h5py
import numpy as np

from emergent.salmon_abm.simulation import simulation


def test_simulation_creates_hdf5_and_populates_agents(tmp_path):
    # initialize a tiny simulation
    sim = simulation(
        model_dir=str(tmp_path),
        model_name="test",
        crs=None,
        basin=None,
        water_temp=10.0,
        start_polygon=None,
        env_files=None,
        longitudinal_profile=None,
        fish_length=None,
        num_timesteps=1,
        num_agents=5,
        use_gpu=False,
        pid_tuning=False,
    )

    # ensure db file exists
    assert os.path.exists(sim.db_path)

    # open HDF5 and validate datasets
    with h5py.File(sim.db_path, "r") as f:
        for ds in ("sex", "length", "weight", "body_depth", "X", "Y"):
            assert ds in f
            data = f[ds][:]
            assert data.shape[0] == 5

    # in-memory arrays should exist and be same length
    assert sim.sex.shape[0] == 5
    assert sim.length.shape[0] == 5
    assert sim.weight.shape[0] == 5
    assert sim.body_depth.shape[0] == 5

    # cleanup: remove temporary db
    try:
        os.remove(sim.db_path)
    except Exception:
        pass
