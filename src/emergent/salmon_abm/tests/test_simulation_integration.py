import os
import h5py
import numpy as np
import pytest
import uuid
from pathlib import Path

from emergent.salmon_abm.simulation import simulation, probe_numba_cuda_runtime


def _unique_h5_path(stem: str) -> Path:
    return Path.cwd() / f"{stem}_{uuid.uuid4().hex}.h5"


def _canonical_env_inputs():
    base = Path.cwd() / "data" / "salmon_abm"
    keys = ["depth.tif", "vel_x.tif", "vel_y.tif", "vel_mag.tif", "vel_dir.tif"]
    env_files = [base / k for k in keys]
    start_polygon = base / "start_loc_river_right.shp"
    missing = [str(p) for p in env_files + [start_polygon] if not p.exists()]
    if missing:
        pytest.skip(f"Missing canonical salmon test inputs: {missing}")
    return [str(p) for p in env_files], str(start_polygon)


def test_simulation_creates_hdf5_and_populates_agents():
    # initialize a tiny simulation
    env_files, start_polygon = _canonical_env_inputs()
    db_file = _unique_h5_path("sim_test_db")
    sim = simulation(
        model_dir=str(Path.cwd()),
        model_name="test",
        crs=None,
        basin=None,
        water_temp=10.0,
        start_polygon=start_polygon,
        env_files=env_files,
        longitudinal_profile=None,
        fish_length=None,
        num_timesteps=1,
        num_agents=5,
        use_gpu=False,
        pid_tuning=False,
        db_path=str(db_file),
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

    # cleanup by calling close (will remove file only if simulation created it internally)
    sim.close()
    # if caller provided db_path, `close()` won't remove it — remove here
    try:
        os.remove(str(db_file))
    except Exception:
        pass


def test_probe_numba_cuda_runtime_has_expected_fields():
    diag = probe_numba_cuda_runtime()
    required = (
        'backend',
        'python_version',
        'numba_installed',
        'numba_version',
        'numba_cuda_importable',
        'cuda_available',
        'device_count',
        'device_name',
        'compute_capability',
        'ready',
        'issues',
    )
    for key in required:
        assert key in diag
    assert isinstance(diag['issues'], list)
    assert isinstance(diag['ready'], bool)


def test_simulation_use_gpu_raises_when_cuda_not_ready():
    diag = probe_numba_cuda_runtime()
    if bool(diag.get('ready', False)):
        pytest.skip('CUDA runtime is available on this host; fail-path test not applicable')

    db_file = _unique_h5_path("sim_test_gpu_gate_db")
    with pytest.raises(RuntimeError, match="GPU mode requested"):
        simulation(
            model_dir=str(Path.cwd()),
            model_name="test_gpu_gate",
            crs=None,
            basin=None,
            water_temp=10.0,
            start_polygon=None,
            env_files=None,
            longitudinal_profile=None,
            fish_length=None,
            num_timesteps=1,
            num_agents=5,
            use_gpu=True,
            pid_tuning=False,
            db_path=str(db_file),
        )
