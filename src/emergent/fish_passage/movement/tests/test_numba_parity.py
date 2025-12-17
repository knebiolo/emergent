import pytest
import numpy as np

def _has_numba():
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _has_numba(), reason="numba not available")
def test_swim_core_numba_vs_numpy():
    from emergent.fish_passage.movement import swim_core as swim_selector
    # swim_selector will pick numba implementation; import numpy fallback directly
    from emergent.fish_passage.movement.swim_core_numpy import swim_core as swim_numpy
    from emergent.fish_passage.movement.swim_core_numba import swim_core as swim_numba

    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    headings = np.array([0.0, np.pi/2])
    speeds = np.array([1.0, 2.0])
    env = np.zeros((2,2), dtype=float)

    np_pos, np_spd = swim_numpy(positions, headings, speeds, env, 1.0)
    nb_pos, nb_spd = swim_numba(positions, headings, speeds, env, 1.0)

    assert np.allclose(np_pos, nb_pos)
    assert np.allclose(np_spd, nb_spd)

@pytest.mark.skipif(not _has_numba(), reason="numba not available")
def test_drag_and_battery_numba_vs_numpy():
    from emergent.fish_passage.movement.drag_and_battery_numpy import drag_and_battery as drag_np
    from emergent.fish_passage.movement.drag_and_battery_numba import drag_and_battery as drag_nb
    positions = np.zeros((3,2), dtype=float)
    speeds = np.array([0.5, 1.0, 2.0])
    headings = np.zeros(3, dtype=float)
    env = np.zeros((3,2), dtype=float)

    f_np, b_np = drag_np(positions, speeds, headings, env, 1.0)
    f_nb, b_nb = drag_nb(positions, speeds, headings, env, 1.0)

    assert np.allclose(f_np, f_nb)
    assert np.allclose(b_np, b_nb)


@pytest.mark.skipif(not _has_numba(), reason="numba not available")
def test_calc_battery_numba_vs_numpy():
    from emergent.fish_passage.movement.calc_battery_numpy import calc_battery as calc_np
    from emergent.fish_passage.movement.calc_battery_numba import calc_battery as calc_nb
    speeds = np.array([0.1, 1.0, 2.0])
    efforts = np.array([1.0, 0.5, 0.2])
    out_np = calc_np(speeds, efforts, 1.0)
    out_nb = calc_nb(speeds, efforts, 1.0)
    assert np.allclose(out_np, out_nb)


@pytest.mark.skipif(not _has_numba(), reason="numba not available")
def test_merged_battery_numba_vs_numpy():
    from emergent.fish_passage.movement.merged_battery_numpy import merged_battery as mb_np
    from emergent.fish_passage.movement.merged_battery_numba import merged_battery as mb_nb
    battery = np.array([1.0, 0.8, 0.5])
    effort = np.array([1.0, 0.5, 0.2])
    speed = np.array([0.1, 1.0, 2.0])
    out_np = mb_np({'battery': battery, 'effort': effort, 'speed': speed}, 1.0)
    out_nb = mb_nb(battery, effort, speed, 1.0)
    assert np.allclose(out_np, out_nb)


@pytest.mark.skipif(not _has_numba(), reason="numba not available")
def test_bout_and_fatigue_numba_vs_numpy():
    from emergent.fish_passage.movement.bout_distance_numpy import bout_distance as bout_np
    from emergent.fish_passage.movement.bout_distance_numba import bout_distance as bout_nb
    from emergent.fish_passage.movement.time_to_fatigue_numpy import time_to_fatigue as ttf_np
    from emergent.fish_passage.movement.time_to_fatigue_numba import time_to_fatigue as ttf_nb
    speeds = np.array([0.5, 1.0, 2.0])
    bd_np = bout_np(speeds, {'mean_duration': 1.0})
    bd_nb = bout_nb(speeds, 1.0)
    assert np.allclose(bd_np, bd_nb)
    energy = np.array([1.0, 2.0, 0.5])
    t_np = ttf_np(energy, {'rate': 0.2})
    t_nb = ttf_nb(energy, 0.2)
    assert np.allclose(t_np, t_nb)
