import numpy as np

from emergent.salmon_abm.movement import movement


class DummySim:
    pass


def test_find_z_shallow_and_deep():
    sim = DummySim()
    sim.depth = np.array([0.01, 2.0])
    sim.body_depth = np.array([10.0, 100.0])
    sim.too_shallow = 0.5

    mv = movement(sim)
    mv.find_z()

    # first fish: depth < body_depth*3/100 -> depth + too_shallow
    assert np.isclose(sim.z[0], sim.depth[0] + sim.too_shallow)
    # second fish: depth < body_depth*3/100 -> depth + too_shallow
    assert np.isclose(sim.z[1], sim.depth[1] + sim.too_shallow)
