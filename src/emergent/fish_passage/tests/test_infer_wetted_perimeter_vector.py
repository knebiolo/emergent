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

