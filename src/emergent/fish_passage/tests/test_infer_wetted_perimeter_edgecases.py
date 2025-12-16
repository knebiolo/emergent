import numpy as np
import h5py
from pathlib import Path
from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


def test_infer_wetted_perimeter_no_wetted(tmp_path: Path):
    """If no depth values exceed the threshold, function should return None."""
    plan = create_minimal_plan(tmp_path / 'plan_none.h5')
    # write a depth timeseries of zeros into a common Results path
    with h5py.File(plan, 'a') as f:
        grp = f.require_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
        depths = np.zeros((1, 4), dtype='f4')
        grp.create_dataset('Cell Hydraulic Depth', data=depths)

    res = infer_wetted_perimeter_from_hecras(str(plan), depth_threshold=0.01)
    assert res is None

