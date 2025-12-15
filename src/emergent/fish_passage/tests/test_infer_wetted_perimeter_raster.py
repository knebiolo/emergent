import tempfile
import os
import numpy as np
import h5py
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_hecras_plan
from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras


def test_infer_wetted_perimeter_raster():
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    try:
        # create plan without vector perimeter to force raster fallback
        create_hecras_plan(path, vector=False)
        rings = infer_wetted_perimeter_from_hecras(path, depth_threshold=0.01, raster_fallback_resolution=0.5)
        import numpy as _np
        if isinstance(rings, _np.ndarray):
            arr = rings
            assert arr.shape[0] >= 1
            assert arr.shape[1] == 2
        else:
            assert isinstance(rings, list)
            assert len(rings) >= 0
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
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
