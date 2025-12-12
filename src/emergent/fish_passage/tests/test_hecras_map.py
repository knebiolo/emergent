import numpy as np
from emergent.fish_passage.io import HECRASMap
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import make_minimal_plan


def test_hecras_map_nearest(tmp_path):
    plan = make_minimal_plan(str(tmp_path / 'plan.h5'))
    m = HECRASMap(plan, 'Cell Hydraulic Depth/Values')
    pts = np.array([[1.0, 0.0], [16.0, 0.0]])
    out = m.map_idw(pts, k=1)
    assert out.shape == (2,)
    # nearest values: [0.0, 0.2]
    assert np.allclose(out, np.array([0.0, 0.2]), atol=1e-6)


def test_hecras_map_idw_weights(tmp_path):
    # Create coords in a line with values increasing
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    values = np.array([[0.0], [1.0], [2.0]])
    plan = make_minimal_plan(str(tmp_path / 'plan2.h5'), coords=coords, values=values)
    m = HECRASMap(plan, 'Cell Hydraulic Depth/Values')
    # query point at 1.0 should give close to 1.0
    pts = np.array([[1.0, 0.0]])
    out = m.map_idw(pts, k=3)
    assert out.shape == (1,)
    assert abs(out[0] - 1.0) < 1e-6
