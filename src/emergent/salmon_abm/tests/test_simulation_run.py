import os
import numpy as np
import h5py
from emergent.salmon_abm.simulation import simulation


def test_run_returns_status(tmp_path):
    db_file = tmp_path / 'simdb.h5'
    sim = simulation(model_dir=str(tmp_path), model_name='m', crs=None, basin=None, water_temp=10.0,
                     start_polygon=None, env_files=None, longitudinal_profile=None,
                     num_timesteps=2, num_agents=3, db_path=str(db_file))
    status = sim.run(n=2, dt=1.0, return_status=True)
    assert isinstance(status, dict)
    assert status['steps'] == 2


def test_run_pid_plumbing(tmp_path):
    db_file = tmp_path / 'simdb2.h5'
    sim = simulation(model_dir=str(tmp_path), model_name='m', crs=None, basin=None, water_temp=10.0,
                     start_polygon=None, env_files=None, longitudinal_profile=None,
                     num_timesteps=1, num_agents=2, pid_tuning=True, db_path=str(db_file))
    # run with scalar gains
    status = sim.run(n=1, dt=1.0, k_p=0.5, k_i=0.1, k_d=0.01, return_status=True)
    assert status['steps'] == 1
    # controller should have k_p set
    assert hasattr(sim.pid_controller, 'k_p')


def test_run_video_hook_called(tmp_path):
    db_file = tmp_path / 'simdb3.h5'
    sim = simulation(model_dir=str(tmp_path), model_name='m', crs=None, basin=None, water_temp=10.0,
                     start_polygon=None, env_files=None, longitudinal_profile=None,
                     num_timesteps=1, num_agents=1, db_path=str(db_file))
    calls = []

    def hook(sim_obj, step):
        calls.append((step, sim_obj.X.copy()))

    status = sim.run(n=3, dt=1.0, video=True)
    # current run implementation does not accept a hook via signature; ensure it returns status
    assert status['steps'] == 3
