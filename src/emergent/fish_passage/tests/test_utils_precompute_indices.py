import numpy as np
from affine import Affine
from emergent.fish_passage.utils import precompute_pixel_indices


class DummySim:
    pass


def test_precompute_pixel_indices_basic():
    sim = DummySim()
    # create a simple 3x3 grid of agent positions
    xs = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    ys = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    sim.X = xs
    sim.Y = ys
    # affine: origin at (0,3), pixel size 1
    sim.depth_rast_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)

    cache = precompute_pixel_indices(sim, mapping_keys={'depth': 'depth_rast_transform'})
    assert 'depth' in cache
    rows, cols = cache['depth']
    assert rows.shape == xs.shape
    assert cols.shape == ys.shape
    assert rows.dtype == np.int32
    assert cols.dtype == np.int32
