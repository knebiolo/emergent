import numpy as np
from affine import Affine
from emergent.fish_passage import geometry
from emergent.fish_passage.geometry import geo_to_pixel_from_inv
from emergent.fish_passage.utils import get_inv_transform


def test_pixel_to_geo_and_back_scalar():
    a = Affine(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    r, c = 5, 7
    x, y = geometry.pixel_to_geo(a, r, c)
    rr, cc = geometry.geo_to_pixel(a, x, y)
    assert int(rr) == r
    assert int(cc) == c
    inv = get_inv_transform(None, a)
    r3, c3 = geo_to_pixel_from_inv(inv, x, y)
    assert int(r3) == r
    assert int(c3) == c


def test_pixel_to_geo_and_back_array():
    a = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    rows = np.array([0, 10, 50])
    cols = np.array([0, 5, 20])
    xs, ys = geometry.pixel_to_geo(a, rows, cols)
    r2, c2 = geometry.geo_to_pixel(a, xs, ys)
    assert np.all(r2 == rows)
    assert np.all(c2 == cols)
    inv = get_inv_transform(None, a)
    r3, c3 = geo_to_pixel_from_inv(inv, xs, ys)
    assert np.array_equal(r2, r3)
    assert np.array_equal(c2, c3)


def test_roundtrip_with_nonzero_origin():
    a = Affine(0.5, 0.0, -100.0, 0.0, -0.5, 200.0)
    rows = np.arange(10)
    cols = np.arange(10)
    xs, ys = geometry.pixel_to_geo(a, rows, cols)
    r2, c2 = geometry.geo_to_pixel(a, xs, ys)
    assert np.all(r2 == rows)
    assert np.all(c2 == cols)
    inv = get_inv_transform(None, a)
    r3, c3 = geo_to_pixel_from_inv(inv, xs, ys)
    assert np.array_equal(r2, r3)
    assert np.array_equal(c2, c3)
