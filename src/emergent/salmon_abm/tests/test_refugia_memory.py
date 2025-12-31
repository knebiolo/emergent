import numpy as np

from emergent.salmon_abm import hdf5_io
from emergent.salmon_abm.simulation import simulation


def test_derive_environment_refugia_uses_fatigued_capacity_and_ref_length(tmp_path):
    sim = simulation(
        model_dir=str(tmp_path),
        model_name="test_refugia",
        crs=None,
        basin="",
        water_temp=10.0,
        start_polygon=None,
        env_files=[],
        longitudinal_profile=None,
        num_timesteps=1,
        num_agents=1,
        db_path=str(tmp_path / "sim.h5"),
    )

    vel_x = np.array(
        [
            [0.2, 0.6, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.6, 0.2],
        ],
        dtype=float,
    )
    vel_y = np.zeros_like(vel_x)
    hdf5_io.write_dataset(sim.db, "environment/vel_x", vel_x)
    hdf5_io.write_dataset(sim.db, "environment/vel_y", vel_y)

    # 0.5 BL/s * 1.0 m body length => 0.5 m/s threshold
    sim.max_s_U_fatigued = np.array([0.5], dtype=np.float32)
    ok = sim.derive_environment_refugia(ref_length_mm=1000.0)
    assert ok is True

    refugia = np.asarray(hdf5_io.read_dataset(sim.db, "environment/refugia"))
    assert refugia.shape == vel_x.shape
    # Cells with vel_mag=0.2 should be refugia; vel_mag=0.6 should not.
    assert int(refugia[0, 0]) == 1
    assert int(refugia[0, 1]) == 0


def test_update_avoid_memory_writes_timestamp(tmp_path):
    sim = simulation(
        model_dir=str(tmp_path),
        model_name="test_memory",
        crs=None,
        basin="",
        water_temp=10.0,
        start_polygon=None,
        env_files=[],
        longitudinal_profile=None,
        num_timesteps=1,
        num_agents=1,
        db_path=str(tmp_path / "sim.h5"),
    )

    depth = np.full((10, 10), 5.0, dtype=np.float32)
    hdf5_io.write_dataset(sim.db, "environment/depth", depth)
    sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

    # Default behavior is sparse avoid history (no dense HDF5 rasters).
    sim.use_sparse_avoid_memory = True
    assert sim.initialize_mental_map(avoid_cell_size=1.0, create_datasets=False) is True

    sim.X = np.array([2.0], dtype=float)
    sim.Y = np.array([-3.0], dtype=float)

    assert sim.update_avoid_memory(t=15.0) is True

    assert sim.avoid_hist_rows is not None
    assert sim.avoid_hist_cols is not None
    assert sim.avoid_hist_t is not None
    assert int(sim.avoid_hist_rows[0, 0]) == 3
    assert int(sim.avoid_hist_cols[0, 0]) == 2
    assert float(sim.avoid_hist_t[0, 0]) == 15.0


def test_update_avoid_memory_persists_dense_when_enabled(tmp_path):
    sim = simulation(
        model_dir=str(tmp_path),
        model_name="test_memory_dense",
        crs=None,
        basin="",
        water_temp=10.0,
        start_polygon=None,
        env_files=[],
        longitudinal_profile=None,
        num_timesteps=1,
        num_agents=1,
        db_path=str(tmp_path / "sim.h5"),
    )

    depth = np.full((10, 10), 5.0, dtype=np.float32)
    hdf5_io.write_dataset(sim.db, "environment/depth", depth)
    sim.depth_rast_transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    sim.avoid_cell_size = 1.0
    sim.use_sparse_avoid_memory = False
    sim.persist_avoid_memory_hdf5 = True

    sim.X = np.array([2.0], dtype=float)
    sim.Y = np.array([-3.0], dtype=float)

    assert sim.update_avoid_memory(t=15.0) is True

    mmap = np.asarray(hdf5_io.read_dataset(sim.db, "memory/0"))
    assert mmap.shape[0] >= 4 and mmap.shape[1] >= 3
    assert float(mmap[3, 2]) == 15.0
