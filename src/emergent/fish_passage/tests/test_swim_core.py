import numpy as np
from emergent.fish_passage import swim_core


def _legacy_swim_core(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt):
    try:
        from emergent.salmon_abm.sockeye import _swim_core_numba as legacy
        return legacy(np.asarray(fv0x), np.asarray(fv0y), np.asarray(accx), np.asarray(accy), np.asarray(pidx), np.asarray(pidy), np.asarray(tired_mask), np.asarray(dead_mask), np.asarray(mask), float(dt))
    except Exception:
        vx = np.asarray(fv0x) + np.asarray(accx) * dt
        vy = np.asarray(fv0y) + np.asarray(accy) * dt
        vx = np.where(~np.asarray(tired_mask), vx + np.asarray(pidx), vx)
        vy = np.where(~np.asarray(tired_mask), vy + np.asarray(pidy), vy)
        vx = np.where((~np.asarray(mask)) | np.asarray(dead_mask), 0.0, vx)
        vy = np.where((~np.asarray(mask)) | np.asarray(dead_mask), 0.0, vy)
        return np.stack((vx * dt, vy * dt), axis=1)


def test_swim_core_basic():
    fv0x = np.array([0.0, 1.0, -1.0, 0.5])
    fv0y = np.array([0.0, 0.5, -0.5, 1.0])
    accx = np.array([0.1, 0.0, -0.1, 0.2])
    accy = np.array([0.0, -0.1, 0.05, -0.2])
    pidx = np.array([0.01, 0.02, -0.02, 0.0])
    pidy = np.array([0.0, 0.01, 0.0, -0.01])
    tired_mask = np.array([False, True, False, False])
    dead_mask = np.array([False, False, True, False])
    mask = np.array([True, True, True, False])
    dt = 0.5

    expected = _legacy_swim_core(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)
    got = swim_core.swim_core(fv0x, fv0y, accx, accy, pidx, pidy, tired_mask, dead_mask, mask, dt)
    assert np.allclose(expected, got)
