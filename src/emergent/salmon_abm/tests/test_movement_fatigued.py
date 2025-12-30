import numpy as np

from emergent.salmon_abm.movement import movement


def _make_sim(
    *,
    length_mm: float,
    swim_behav: int,
    max_s_u_bl_s: float,
    max_s_u_fatigued_bl_s: float,
):
    class Sim:
        pass

    sim = Sim()
    sim.num_agents = 1
    sim.length = np.array([length_mm], dtype=float)
    sim.x_vel = np.array([0.0], dtype=float)
    sim.y_vel = np.array([0.0], dtype=float)
    sim.ideal_sog = np.array([0.0], dtype=float)
    sim.heading = np.array([0.0], dtype=float)
    sim.swim_behav = np.array([swim_behav], dtype=int)
    sim.max_s_U = np.array([max_s_u_bl_s], dtype=float)
    sim.max_s_U_fatigued = np.array([max_s_u_fatigued_bl_s], dtype=float)
    sim.wave_drag = np.array([1.0], dtype=float)
    sim.water_temp = 10.0
    sim.drag_coeff = lambda re: np.ones_like(re, dtype=float)
    return sim


def test_ideal_drag_fun_caps_by_length_scaled_max_s_u_when_refugia_mode():
    sim = _make_sim(length_mm=500.0, swim_behav=2, max_s_u_bl_s=2.0, max_s_u_fatigued_bl_s=1.0)
    mv = movement(sim)

    # Large requested over-ground velocity; should be capped to max_s_U (BL/s) * length (m)
    fish_vel = np.array([[10.0, 0.0]], dtype=float)
    mv.ideal_drag_fun(fish_velocities=fish_vel)

    # 2.0 BL/s * 0.5 m = 1.0 m/s
    assert np.isclose(sim.max_practical_sog[0, 0], 1.0)


def test_ideal_drag_fun_caps_more_when_holding_mode_uses_fatigued_capacity():
    sim = _make_sim(length_mm=500.0, swim_behav=3, max_s_u_bl_s=2.0, max_s_u_fatigued_bl_s=1.0)
    mv = movement(sim)

    fish_vel = np.array([[10.0, 0.0]], dtype=float)
    mv.ideal_drag_fun(fish_velocities=fish_vel)

    # 1.0 BL/s * 0.5 m = 0.5 m/s
    assert np.isclose(sim.max_practical_sog[0, 0], 0.5)

