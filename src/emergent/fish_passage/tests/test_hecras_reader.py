import h5py
import numpy as np
from pathlib import Path
from emergent.fish_passage.hecras_reader import load_hecras_cells


def test_load_hecras_cells(tmp_path: Path):
    p = tmp_path / 'plan.h5'
    coords = np.column_stack((np.linspace(0,1,10), np.linspace(0,1,10)))
    vals = np.linspace(0,1,10)
    with h5py.File(str(p), 'w') as hdf:
        hdf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        hdf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([vals]))
    c, v = load_hecras_cells(str(p))
    assert c.shape[0] == 10
    assert v.shape[0] == 10
 