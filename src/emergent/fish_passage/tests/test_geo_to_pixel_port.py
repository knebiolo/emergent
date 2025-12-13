import numpy as np
from affine import Affine
from emergent.fish_passage.geometry import geo_to_pixel


def test_geo_to_pixel_basic():
    # Create a simple affine: cell size 1.0, origin at (0, 10) with negative y
    aff = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
    xs = np.array([0.5, 2.5, 5.0])
    ys = np.array([9.5, 7.5, 5.0])
    rows, cols = geo_to_pixel(aff, xs, ys)
    assert np.array_equal(rows, np.array([0, 2, 5]))
    assert np.array_equal(cols, np.array([0, 2, 5]))


def test_geo_to_pixel_scalar():
    aff = Affine(2.0, 0.0, -1.0, 0.0, -2.0, 9.0)
    r, c = geo_to_pixel(aff, 3.0, 5.0)
    assert isinstance(r, (int, np.integer))
    assert isinstance(c, (int, np.integer))
