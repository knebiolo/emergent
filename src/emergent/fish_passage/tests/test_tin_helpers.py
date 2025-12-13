import numpy as np
from shapely.geometry import Polygon
from emergent.fish_passage.tin_helpers import alpha_shape, sample_evenly


def test_sample_evenly_small():
    pts = np.arange(20).reshape((10,2)).astype(float)
    out = sample_evenly(pts, 5)
    assert out.shape[0] == 5


def test_alpha_shape_convex():
    # rectangle
    xs, ys = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
    pts = np.column_stack((xs.flatten(), ys.flatten()))
    poly = alpha_shape(pts, alpha=0.5)
    assert poly is not None
    assert isinstance(poly, Polygon)
 