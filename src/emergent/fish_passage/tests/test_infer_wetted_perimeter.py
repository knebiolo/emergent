import numpy as np
from pathlib import Path
from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures.hdf5_plan_fixture import create_minimal_plan


def test_infer_wetted_perimeter_minimal(tmp_path: Path):
    p = tmp_path / 'plan.h5'
    create_minimal_plan(p)
    # augment file with minimal vector geometry datasets used by vector method
    import h5py
    import numpy as np
    with h5py.File(str(p), 'a') as f:
        # facepoints: simple four facepoints at corners
        facepoints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype='f4')
        f.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Coordinate', data=facepoints)
        # mark all facepoints as perimeter (-1)
        f.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter', data=np.array([-1, -1, -1, -1], dtype='i4'))
        # cells face info: for 4 cells, each references 4 facepoints (start index, count)
        # here we'll point each cell to a sequence in a flattened face index list
        # simple mapping: 4 cells each with faces starting at 0,1,2,3 and count 1 (toy)
        fi = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype='i4')
        f.create_dataset('Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info', data=fi)
        # perimeter coordinates: repeat facepoints
        f.create_dataset('Geometry/2D Flow Areas/2D area/Perimeter', data=facepoints)
        # create the long-form depth dataset expected by the vector method
        # shape: (1, n_nodes)
        depths = np.array([[0.1, 0.2, 0.3, 0.4]], dtype='f4')
        long_path = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'
        # ensure groups exist
        grp = f.require_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
        grp.create_dataset('Cell Hydraulic Depth', data=depths)
    rings = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.05)
    assert isinstance(rings, list)
    # minimal plan has depths [0.1,0.2,0.3,0.4] so wetted -> non-empty
    assert len(rings) >= 1
    for r in rings:
        assert hasattr(r, 'shape')
        assert r.shape[1] == 2
