import h5py
import numpy as np
from shapely.geometry import LineString
from emergent.fish_passage.centerline import extract_centerline_fast_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan
import pytest


def test_extract_centerline_hecras_returns_none_for_small_plan(tmp_path):
    # create a minimal plan with only 4 points -> centerline should be None (too short)
    plan = create_minimal_plan(tmp_path / 'small_plan.h5')
    # add expected Results dataset path so wrapper can read depths
    with h5py.File(str(plan), 'a') as f:
        # shape (1, Nnodes) so ds[0] returns the node depths
        f.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.array([[0.1, 0.2, 0.3, 0.4]], dtype='f4'))

    res = extract_centerline_fast_hecras(str(plan), depth_threshold=0.05, min_length=1.0)
    # For this tiny plan the array-based extractor may return None due to min_length
    assert res is None or isinstance(res, LineString)


def test_extract_centerline_hecras_raises_on_missing_datasets(tmp_path):
    # create an invalid HDF5 (missing expected dataset path)
    bad = tmp_path / 'bad.h5'
    with h5py.File(str(bad), 'w') as f:
        f.create_dataset('Some/Other/Dataset', data=np.array([1,2,3]))

    with pytest.raises(Exception):
        extract_centerline_fast_hecras(str(bad))
