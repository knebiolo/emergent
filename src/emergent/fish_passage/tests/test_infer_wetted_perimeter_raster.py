import numpy as np
import h5py
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


def test_infer_wetted_perimeter_raster_fallback(tmp_path):
    plan = create_minimal_plan(tmp_path / 'plan_raster.h5')
    # overwrite coords with a 5x5 grid to force raster fallback
    with h5py.File(plan, 'a') as f:
        coords = []
        for i in range(5):
            for j in range(5):
                coords.append([float(i), float(j)])
        coords = np.array(coords, dtype='f4')
        # replace the coords dataset to match new (25,2) shape
        dpath = 'Geometry/2D Flow Areas/2D area/Cells Center Coordinate'
        if dpath in f:
            del f[dpath]
        f.create_dataset(dpath, data=coords)
        f.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.array([np.ones(len(coords), dtype='f4')]))
    perim = infer_wetted_perimeter_from_hecras(str(plan), depth_threshold=0.05, raster_fallback_resolution=1.0, verbose=False)
    assert perim is not None
    assert isinstance(perim, np.ndarray)
    assert perim.shape[1] == 2
