import numpy as np
import h5py

from emergent.fish_passage.io import HECRASMap


def test_hecrasmap_single_field(tmp_path):
    p = tmp_path / "plan.h5"
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    field = np.array([10.0, 20.0, 30.0, 40.0])
    with h5py.File(p, 'w') as f:
        grp = f.require_group('Geometry/2D Flow Areas/2D area')
        grp.create_dataset('Cells Center Coordinate', data=coords)
        # place results under a Results/ group so dataset discovery picks it
        rgrp = f.require_group('Results/SomeField')
        rgrp.create_dataset('Values', data=field)

    # pass field as a single-string (legacy behavior should return ndarray)
    m = HECRASMap(str(p), field_names='SomeField')
    out = m.map_idw([coords[0]], k=1)
    assert np.isclose(out, 10.0).all()


def test_hecrasmap_list_field_returns_dict(tmp_path):
    p = tmp_path / "plan2.h5"
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    f1 = np.array([5.0, 6.0])
    with h5py.File(p, 'w') as f:
        grp = f.require_group('Geometry/2D Flow Areas/2D area')
        grp.create_dataset('Cells Center Coordinate', data=coords)
        f.create_dataset('SomeField', data=f1)

    m = HECRASMap(str(p), field_names=['SomeField'])
    out = m.map_idw([[1.0, 0.0]], k=1)
    assert isinstance(out, dict)
    assert 'SomeField' in out
    assert np.isclose(out['SomeField'][0], 6.0)
