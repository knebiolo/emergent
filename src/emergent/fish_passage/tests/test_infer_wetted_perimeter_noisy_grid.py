import numpy as np
import h5py
from pathlib import Path
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import make_minimal_plan


def test_infer_wetted_perimeter_noisy_grid(tmp_path: Path):
    path = tmp_path / 'plan_noisy.h5'
    make_minimal_plan(path, coords=np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0],[3.0,0.0],[4.0,0.0]]), values=np.array([[0.0],[1.0],[0.5],[0.8],[0.0]]))
    # the simple 1D coords simulate a narrow wetted channel; expect None or a small array
    perim = infer_wetted_perimeter_from_hecras(str(path), depth_threshold=0.05, raster_fallback_resolution=0.5)
    # ensure function returns either None or an array-like perimeter
    if perim is None:
        assert perim is None
    else:
        assert hasattr(perim, 'shape')
        assert perim.shape[1] == 2

