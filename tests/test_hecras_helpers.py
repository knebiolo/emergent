import os
import tempfile
import h5py
import numpy as np

from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras


def make_synthetic_hecras_hdf(path, n_cells=100, times=3):
    # simple grid coords and a depth timeseries with varying wetted cells
    coords = np.column_stack((np.linspace(0, 9, int(np.sqrt(n_cells))).repeat(int(np.sqrt(n_cells))),
                              np.tile(np.linspace(0, 9, int(np.sqrt(n_cells))), int(np.sqrt(n_cells)))))
    with h5py.File(path, 'w') as hdf:
        grp_geom = hdf.create_group('Geometry/2D Flow Areas/2D area')
        grp_geom.create_dataset('Cells Center Coordinate', data=coords)
        # Minimal facepoints/perimeter to exercise vector method may be absent; tests should accept raster fallback
        # Write a Results depth timeseries dataset: shape (times, n_cells)
        grp_res = hdf.create_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
        depths = np.zeros((times, len(coords)), dtype=np.float32)
        for t in range(times):
            # progressively flood more cells
            depths[t, : int(len(coords) * (0.2 + 0.2 * t))] = 0.2 + 0.1 * t
        grp_res.create_dataset('Cell Hydraulic Depth', data=depths)


def test_infer_wetted_perimeter_basic(tmp_path):
    p = tmp_path / 'synthetic_hecras.h5'
    make_synthetic_hecras_hdf(str(p), n_cells=100, times=4)

    # timestep 0
    perims0 = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.1, raster_fallback_resolution=2.0, verbose=False, timestep=0)
    assert isinstance(perims0, list)
    assert len(perims0) >= 1

    # later timestep (more flooded)
    perims2 = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.1, raster_fallback_resolution=2.0, verbose=False, timestep=2)
    assert isinstance(perims2, list)
    assert len(perims2) >= 1
