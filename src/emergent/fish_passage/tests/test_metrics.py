import numpy as np
from emergent.fish_passage.metrics import compute_schooling_metrics_biological, compute_drafting_benefits


class DummyBW:
    def __init__(self):
        self.cohesion_radius_relaxed = 2.0
        self.separation_radius = 1.0
        self.drafting_angle_tolerance = 30.0
        self.drag_reduction_single = 0.15
        self.drag_reduction_dual = 0.25


def test_compute_schooling_metrics_empty():
    bw = DummyBW()
    out = compute_schooling_metrics_biological(np.zeros((0,2)), np.zeros((0,)), np.zeros((0,)), bw)
    assert out['overall_schooling'] == 0.0


def test_compute_schooling_metrics_simple():
    # Two agents close together aligned
    pos = np.array([[0.0, 0.0], [0.5, 0.0]])
    headings = np.array([0.0, 0.0])
    bl = np.array([1.0, 1.0])
    bw = DummyBW()
    out = compute_schooling_metrics_biological(pos, headings, bl, bw)
    assert 0.0 <= out['cohesion_score'] <= 1.0
    assert -1.0 <= out['alignment_score'] <= 1.0


def test_compute_drafting_benefits_simple():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    headings = np.array([0.0, 0.0, 0.0])
    velocities = np.array([0.0, 0.0, 0.0])
    bl = np.array([1.0, 1.0, 1.0])
    bw = DummyBW()
    reductions = compute_drafting_benefits(pos, headings, velocities, bl, bw)
    assert reductions.shape == (3,)
    # agent 0 has agent 1 ahead -> receives a reduction
    assert reductions[0] >= 0.0
    # middle agent may have a benefit
    assert reductions[1] >= 0.0
    # agent 2 has no one ahead -> no reduction
    assert reductions[2] == 0.0
