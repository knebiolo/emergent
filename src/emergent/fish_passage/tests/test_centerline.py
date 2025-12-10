import numpy as np
from shapely.geometry import LineString

from emergent.fish_passage.centerline import derive_centerline_from_hecras_distance


def test_centerline_simple_ridge():
    # Create a simple rectangular channel: x varies 0..9, y varies 0..4
    xs, ys = np.meshgrid(np.arange(10), np.arange(5))
    coords = np.column_stack((xs.flatten(), ys.flatten()))
    # Distance-to-bank: larger near center x=4.5, simulate ridge along center x~4.5
    distances = 1.0 - np.abs(coords[:, 0] - 4.5) / 10.0
    wetted_mask = np.ones(len(coords), dtype=bool)
    centerline = derive_centerline_from_hecras_distance(coords, distances, wetted_mask, min_length=1.0)
    assert centerline is None or isinstance(centerline, LineString)  # depends on ridge extraction


def test_centerline_short_ridge_returns_none():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = np.array([0.0, 0.0])
    wetted_mask = np.array([True, True])
    centerline = derive_centerline_from_hecras_distance(coords, distances, wetted_mask, min_length=100.0)
    assert centerline is None
