from emergent.fish_passage import core


def test_compute_travel_times_basic():
    distances = [0.0, 10.0, 20.0]
    times = core.compute_travel_times(distances, 2.0)
    assert times == [0.0, 5.0, 10.0]


def test_summarize_times():
    times = [3.0, 1.0, 2.0]
    mn, mx = core.summarize_times(times)
    assert mn == 1.0 and mx == 3.0
