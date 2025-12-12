import pytest
from emergent.fish_passage.behavior import BehavioralWeights


def test_behavioral_weights_defaults():
    bw = BehavioralWeights()
    assert bw.cohesion_radius_relaxed == 2.0
    assert bw.separation_radius == 1.0
    assert bw.drafting_angle_tolerance == 30.0
    assert bw.get('nonexistent', 'x') == 'x'

