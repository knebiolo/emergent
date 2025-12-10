import numpy as np
from emergent.fish_passage.geometry import project_points_onto_line


def test_project_points_on_horizontal_line():
    xs = np.array([0.0, 10.0])
    ys = np.array([0.0, 0.0])
    px = np.array([0.0, 5.0, 10.0])
    py = np.array([0.0, 0.0, 0.0])
    d = project_points_onto_line(xs, ys, px, py)
    assert np.allclose(d, [0.0, 5.0, 10.0])


def test_project_points_on_polyline():
    xs = np.array([0.0, 5.0, 5.0])
    ys = np.array([0.0, 0.0, 5.0])
    # point near second segment (vertical)
    px = np.array([5.0, 2.5])
    py = np.array([2.5, 0.0])
    d = project_points_onto_line(xs, ys, px, py)
    # first point projects to midway on second segment: distance = 5 + 2.5 = 7.5
    assert abs(d[0] - 7.5) < 1e-6
    # second point projects to middle of first segment: distance = 2.5
    assert abs(d[1] - 2.5) < 1e-6
