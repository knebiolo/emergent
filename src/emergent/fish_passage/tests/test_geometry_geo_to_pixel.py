import numpy as np
from affine import Affine
from emergent.fish_passage.geometry import compute_affine_from_hecras, pixel_to_geo, geo_to_pixel
from emergent.fish_passage.geometry import geo_to_pixel_from_inv
from emergent.fish_passage.utils import get_inv_transform


def test_geo_pixel_roundtrip_scalar():
    # construct a simple affine: cell size 2, origin at (0, 10)
    a = Affine(2.0, 0.0, 0.0, 0.0, -2.0, 10.0)
    # pick pixel location
    r, c = 3, 4
    x, y = pixel_to_geo(a, r, c)
    rr, cc = geo_to_pixel(a, x, y)
    assert int(rr) == r
    assert int(cc) == c
    inv = get_inv_transform(None, a)
    r2, c2 = geo_to_pixel_from_inv(inv, x, y)
    assert int(r2) == r
    assert int(c2) == c


def test_geo_pixel_roundtrip_vector():
    coords = np.array([[0.0, 0.0], [2.0, -2.0], [4.0, -4.0]])
    # compute affine from coords
    aff = compute_affine_from_hecras(coords)
    # pick rows/cols
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    xs, ys = pixel_to_geo(aff, rows, cols)
    rr, cc = geo_to_pixel(aff, xs, ys)
    assert np.all(rr == rows)
    assert np.all(cc == cols)
    inv = get_inv_transform(None, aff)
    r2, c2 = geo_to_pixel_from_inv(inv, xs, ys)
    assert np.array_equal(rr, r2)
    assert np.array_equal(cc, c2)
