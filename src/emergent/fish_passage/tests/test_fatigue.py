import numpy as np
from emergent.fish_passage import fatigue


def _legacy_calc(battery, per_rec, ttf, mask_sustained, dt):
    try:
        from emergent.salmon_abm.sockeye import _calc_battery_numba as legacy
        return legacy(np.asarray(battery).copy(), np.asarray(per_rec).copy(), np.asarray(ttf).copy(), np.asarray(mask_sustained).copy(), dt)
    except Exception:
        # fallback: use same numpy logic as legacy
        battery = np.asarray(battery, dtype=np.float64).copy()
        per_rec = np.asarray(per_rec, dtype=np.float64)
        ttf = np.asarray(ttf, dtype=np.float64)
        mask_sustained = np.asarray(mask_sustained, dtype=np.bool_)
        battery[mask_sustained] += per_rec[mask_sustained]
        mask_non = ~mask_sustained
        ttf0 = ttf[mask_non] * battery[mask_non]
        ttf1 = ttf0 - dt
        safe = ttf0 != 0
        ratio = np.ones_like(ttf0)
        ratio[safe] = np.maximum(0.0, ttf1[safe] / ttf0[safe])
        battery[mask_non] = battery[mask_non] * ratio
        np.clip(battery, 0.0, 1.0, out=battery)
        return battery


def test_calc_battery_basic():
    battery = np.array([1.0, 0.5, 0.2, 0.0])
    per_rec = np.array([0.1, 0.1, 0.1, 0.1])
    ttf = np.array([10.0, 5.0, 2.0, 1.0])
    mask_sustained = np.array([True, False, False, False])
    dt = 0.5

    expected = _legacy_calc(battery, per_rec, ttf, mask_sustained, dt)
    got = fatigue.calc_battery(battery, per_rec, ttf, mask_sustained, dt)
    assert np.allclose(expected, got)


def test_calc_battery_edge_cases():
    battery = np.array([0.0, 1.0, 0.3, 0.4])
    per_rec = np.array([0.0, 0.2, 0.05, 0.01])
    ttf = np.array([0.0, 1.0, 0.0, 2.0])
    mask_sustained = np.array([False, True, False, False])
    dt = 1.0

    expected = _legacy_calc(battery, per_rec, ttf, mask_sustained, dt)
    got = fatigue.calc_battery(battery, per_rec, ttf, mask_sustained, dt)
    assert np.allclose(expected, got)


def test_merged_battery_parity():
    battery = np.array([0.2, 0.9, 0.0, 0.5, 1.0])
    per_rec = np.array([0.01, 0.1, 0.05, 0.02, 0.0])
    ttf = np.array([1.0, 2.0, 0.0, 5.0, 10.0])
    mask_sustained = np.array([False, True, False, False, False])
    dt = 0.5

    try:
        from emergent.salmon_abm.sockeye import _merged_battery_numba as legacy_merged
        expected = legacy_merged(battery.copy(), per_rec.copy(), ttf.copy(), mask_sustained.copy(), dt)
    except Exception:
        # fallback compute same as legacy numpy fallback
        expected = battery.copy()
        for i in range(expected.size):
            if mask_sustained[i]:
                expected[i] = expected[i] + per_rec[i]
            else:
                t0 = ttf[i] * expected[i]
                if t0 <= 0.0:
                    expected[i] = 0.0
                else:
                    t1 = t0 - dt
                    ratio = t1 / t0
                    if ratio < 0.0:
                        ratio = 0.0
                    expected[i] = expected[i] * ratio
        np.clip(expected, 0.0, 1.0, out=expected)

    got = fatigue.merged_battery(battery, per_rec, ttf, mask_sustained, dt)
    assert np.allclose(expected, got)
