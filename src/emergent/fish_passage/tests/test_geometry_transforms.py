import numpy as np
from affine import Affine
from emergent.fish_passage.geometry import geo_to_pixel, pixel_to_geo, compute_affine_from_hecras


def test_affine_compute_and_roundtrip():
    # Create synthetic clustered coords forming a grid around (0,0)
    xs = np.linspace(0.0, 9.0, 10)
    ys = np.linspace(0.0, 9.0, 10)
    X, Y = np.meshgrid(xs, ys)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    # Compute affine with small target cell size
    aff = compute_affine_from_hecras(coords, target_cell_size=1.0)
    assert isinstance(aff, Affine)

    # Test pixel_to_geo then geo_to_pixel roundtrip on a few points
    rows = np.array([0, 1, 5, 9])
    cols = np.array([0, 2, 5, 9])
    xs_out, ys_out = pixel_to_geo(aff, rows, cols)
    r2, c2 = geo_to_pixel(aff, xs_out, ys_out)
    assert np.all(r2 == rows)
    assert np.all(c2 == cols)


def test_geo_to_pixel_scalars_and_arrays():
    aff = Affine.translation(10.0, 20.0) * Affine.scale(2.0, -2.0)
    # scalar test
    r, c = geo_to_pixel(aff, 12.0, 18.0)
    xr, yc = pixel_to_geo(aff, r, c)
    assert np.isclose(xr, 12.0)
    assert np.isclose(yc, 18.0)

    # array test
    xs = np.array([12.0, 14.0, 16.0])
    ys = np.array([18.0, 14.0, 10.0])
    rs, cs = geo_to_pixel(aff, xs, ys)
    x2, y2 = pixel_to_geo(aff, rs, cs)
    assert np.allclose(x2, xs)
    assert np.allclose(y2, ys)


def test_geo_to_pixel_from_inv_matches_geo_to_pixel_and_cached_inv():
    from affine import Affine
    from emergent.fish_passage.utils import get_inv_transform
    from emergent.fish_passage.geometry import geo_to_pixel_from_inv

    aff = Affine.translation(10.0, 20.0) * Affine.scale(2.0, -2.0)
    # Build inverse via utils cache
    inv1 = get_inv_transform(None, aff)

    xs = np.array([12.0, 14.0, 16.0])
    ys = np.array([18.0, 14.0, 10.0])

    # Direct geo_to_pixel
    rs, cs = geo_to_pixel(aff, xs, ys)
    # Using precomputed inv
    r2, c2 = geo_to_pixel_from_inv(inv1, xs, ys)
    assert np.all(r2 == rs)
    assert np.all(c2 == cs)
