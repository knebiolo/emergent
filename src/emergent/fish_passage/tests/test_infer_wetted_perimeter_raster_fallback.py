import io
import h5py
import numpy as np

from emergent.fish_passage.io import infer_wetted_perimeter_from_hecras


def make_inmemory_hecras():
    bio = io.BytesIO()
    # create an in-memory HDF5 file with core driver
    f = h5py.File(bio, mode='w')

    # add minimal datasets expected by the function
    grp_geom = f.create_group('Geometry/2D Flow Areas/2D area')
    # create simple 3x3 grid of cell centers
    xs = np.linspace(0.0, 20.0, 9)
    ys = np.linspace(0.0, 20.0, 9)
    xv, yv = np.meshgrid(xs, ys)
    centers = np.vstack([xv.ravel(), yv.ravel()]).T
    grp_geom.create_dataset('Cells Center Coordinate', data=centers)
    # No FacePoints Coordinate or Is Perimeter datasets to force raster fallback

    # Create a simple depth series that makes half the cells wetted
    grp_res = f.create_group('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area')
    depth = np.zeros((1, centers.shape[0]), dtype=float)
    # mark central band as wetted
    mid = centers.shape[0] // 2
    depth[0, mid - 4: mid + 5] = 0.2
    grp_res.create_dataset('Cell Hydraulic Depth', data=depth)

    return f


def test_raster_fallback_produces_polygons():
    f = make_inmemory_hecras()
    # use the file object directly
    rings = infer_wetted_perimeter_from_hecras(f, raster_fallback_resolution=5.0)
    assert isinstance(rings, list)
    assert len(rings) >= 1
    # each ring should have at least 4 coords
    assert any(len(r) >= 4 for r in rings)
    f.close()
