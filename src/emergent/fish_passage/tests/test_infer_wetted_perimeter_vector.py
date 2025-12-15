import tempfile
import os
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_hecras_plan
from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras


def test_infer_wetted_perimeter_vector():
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        create_hecras_plan(path, vector=True)
        rings = infer_wetted_perimeter_from_hecras(path, depth_threshold=0.01, timestep=0)
        # function may return a single ndarray (exterior coords) or a list of rings
        import numpy as _np
        if isinstance(rings, _np.ndarray):
            arr = rings
            assert arr.shape[0] >= 1
            assert arr.shape[1] == 2
        else:
            assert isinstance(rings, list)
            assert len(rings) >= 1
            # each ring should be an array of coordinates with at least 3 points
            for ring in rings:
                assert ring.shape[0] >= 3
                assert ring.shape[1] == 2
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
import numpy as np
import h5py
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


def test_infer_wetted_perimeter_vector_basic(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_vector.h5')
    with h5py.File(plan, 'a') as f:
        f.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.array([[0.2,0.2,0.2,0.2]], dtype='f4'))
    perim = infer_wetted_perimeter_from_hecras(str(plan), depth_threshold=0.05)
    assert perim is not None
    assert isinstance(perim, np.ndarray)
    assert perim.shape[1] == 2


def test_infer_wetted_perimeter_vector_no_wetted(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_no_wet.h5')
    with h5py.File(plan, 'a') as f:
        f.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.array([[0.0,0.0,0.0,0.0]], dtype='f4'))
    perim = infer_wetted_perimeter_from_hecras(str(plan), depth_threshold=0.05)
    assert perim is None

