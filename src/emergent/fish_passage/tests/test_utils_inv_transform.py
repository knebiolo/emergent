from affine import Affine
from types import SimpleNamespace
from emergent.fish_passage import utils


def test_get_inv_transform_caches():
    a = Affine(2.0, 0.0, 1.0, 0.0, -2.0, 5.0)
    sim = SimpleNamespace()
    inv1 = utils.get_inv_transform(sim, a)
    inv2 = utils.get_inv_transform(sim, a)
    assert inv1 is inv2


def test_get_inv_transform_sim_none():
    a = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    inv = utils.get_inv_transform(None, a)
    assert inv == ~a
