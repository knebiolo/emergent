import numpy as np


def test_null_writer_noop():
    from emergent.salmon_abm.async_output import NullWriter

    w = NullWriter()
    w.start()
    assert w.submit(0, {"X": np.array([1, 2, 3])}) is True
    assert w.close() is True


def test_thread_hdf_writer_writes_timeseries(tmp_path):
    import h5py

    from emergent.salmon_abm.async_output import AsyncWriteConfig, ThreadHdfWriter

    n = 5
    steps = 4
    path = tmp_path / "async.h5"

    # Pre-create datasets (Phase 1 keeps dataset layout simple; Phase 3 can refine).
    with h5py.File(path, "w") as h5:
        h5.create_dataset("agent_data/X", shape=(n, steps), dtype="float32")
        h5.create_dataset("agent_data/Y", shape=(n, steps), dtype="float32")

    cfg = AsyncWriteConfig(h5_path=str(path), n_agents=n, n_steps=steps, mode="a", queue_max=8, policy="block")
    w = ThreadHdfWriter(cfg)
    w.start()

    for t in range(steps):
        x = np.full((n,), float(t), dtype=np.float32)
        y = np.full((n,), float(100 + t), dtype=np.float32)
        ok = w.submit(t, {"agent_data/X": x, "agent_data/Y": y}, copy=True)
        assert ok is True

    assert w.close(timeout_s=5.0) is True

    with h5py.File(path, "r") as h5:
        X = np.asarray(h5["agent_data/X"])
        Y = np.asarray(h5["agent_data/Y"])

    for t in range(steps):
        assert np.allclose(X[:, t], float(t))
        assert np.allclose(Y[:, t], float(100 + t))


def test_thread_hdf_writer_drop_oldest_does_not_deadlock(tmp_path):
    import h5py

    from emergent.salmon_abm.async_output import AsyncWriteConfig, ThreadHdfWriter

    n = 20
    steps = 50
    path = tmp_path / "async_drop.h5"

    with h5py.File(path, "w") as h5:
        h5.create_dataset("agent_data/X", shape=(n, steps), dtype="float32")

    cfg = AsyncWriteConfig(h5_path=str(path), n_agents=n, n_steps=steps, mode="a", queue_max=2, policy="drop_oldest")
    w = ThreadHdfWriter(cfg)
    w.start()

    for t in range(steps):
        x = np.full((n,), float(t), dtype=np.float32)
        w.submit(t, {"agent_data/X": x}, copy=True)

    assert w.close(timeout_s=5.0) is True


def test_shm_ring_hdf_writer_writes_timeseries(tmp_path):
    import h5py
    import pytest

    try:
        from emergent.salmon_abm.async_output import AsyncWriteConfig, ShmRingHdfWriter
    except Exception as e:
        pytest.skip(f"shared-memory writer unavailable: {e}")

    n = 7
    steps = 6
    path = tmp_path / "async_shm.h5"

    with h5py.File(path, "w") as h5:
        h5.create_dataset("agent_data/X", shape=(n, steps), dtype="float32")
        h5.create_dataset("agent_data/Y", shape=(n, steps), dtype="float32")

    cfg = AsyncWriteConfig(h5_path=str(path), n_agents=n, n_steps=steps, mode="a", queue_max=8, policy="block")
    w = ShmRingHdfWriter(cfg, keys=("agent_data/X", "agent_data/Y"), ring_slots=4, dtype=np.float32)
    w.start()

    for t in range(steps):
        x = np.full((n,), float(t), dtype=np.float32)
        y = np.full((n,), float(100 + t), dtype=np.float32)
        ok = w.submit(t, {"agent_data/X": x, "agent_data/Y": y}, copy=False)
        assert ok is True

    assert w.close(timeout_s=10.0) is True

    with h5py.File(path, "r") as h5:
        X = np.asarray(h5["agent_data/X"])
        Y = np.asarray(h5["agent_data/Y"])

    for t in range(steps):
        assert np.allclose(X[:, t], float(t))
        assert np.allclose(Y[:, t], float(100 + t))
