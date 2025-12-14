import types
from affine import Affine
from emergent.fish_passage.utils import get_inv_transform


class DummySim:
    pass


def test_get_inv_transform_caches_and_returns_inverse():
    sim = DummySim()
    t = Affine.translation(10, 20) * Affine.scale(2, -2)
    inv1 = get_inv_transform(sim, t)
    inv2 = get_inv_transform(sim, t)
    assert inv1 is inv2
    # verify multiplication behavior
    x, y = 5.0, 3.0
    c, r = (inv1 * (x, y))
    # also check ~t equals inv1
    inv_direct = ~t
    inv_c, inv_r = inv_direct * (x, y)
    assert float(c) == float(inv_c)
