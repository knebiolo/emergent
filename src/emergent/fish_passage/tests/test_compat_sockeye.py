import numpy as np
from emergent.fish_passage import compat_sockeye as compat
from emergent.fish_passage import metrics as metrics
from emergent.fish_passage import fatigue as fatigue
from emergent.fish_passage import projection as projection


def test_compute_schooling_and_drafting_match():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    headings = np.array([0.0, 0.0, 0.0])
    lengths = np.array([0.2, 0.2, 0.2])
    class W:  # lightweight weights
        cohesion_radius_relaxed = 2.0
        separation_radius = 1.0
        threat_level = 0.0
        drafting_angle_tolerance = 30.0
        drafting_forward_radius = 2.0
        drag_reduction_single = 0.15
        drag_reduction_dual = 0.25

    w = W()
    out1 = compat.compute_schooling_metrics_biological(pos, headings, lengths, w)
    out2 = metrics.compute_schooling_metrics_biological(pos, headings, lengths, w)
    assert np.isclose(out1['cohesion_score'], out2['cohesion_score'])
    assert np.isclose(out1['alignment_score'], out2['alignment_score'])

    vel = np.zeros_like(pos)
    d1 = compat.compute_drafting_benefits(pos, headings, vel, lengths, w)
    d2 = metrics.compute_drafting_benefits(pos, headings, vel, lengths, w)
    assert np.allclose(d1, d2)


def test_fatigue_and_projection_wrappers():
    battery = np.array([0.5, 0.6, 0.2])
    per_rec = np.array([0.1, 0.0, 0.05])
    ttf = np.array([10.0, 5.0, 8.0])
    mask_sustained = np.array([True, False, False])
    dt = 1.0
    b1 = compat.calc_battery(battery, per_rec, ttf, mask_sustained, dt)
    b2 = fatigue.calc_battery(battery, per_rec, ttf, mask_sustained, dt)
    assert np.allclose(b1, b2)

    mb1 = compat.merged_battery(battery, per_rec, ttf, mask_sustained, dt)
    mb2 = fatigue.merged_battery(battery, per_rec, ttf, mask_sustained, dt)
    assert np.allclose(mb1, mb2)

    px = np.array([0.1, 1.1, 2.0])
    py = np.array([0.0, 0.0, 0.0])
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 0.0, 0.0])
    p1 = compat.project_points_onto_line(xs, ys, px, py)
    p2 = projection.project_points_onto_line(xs, ys, px, py)
    assert np.allclose(p1, p2)
