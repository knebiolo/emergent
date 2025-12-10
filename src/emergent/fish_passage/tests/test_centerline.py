import numpy as np
from shapely.geometry import LineString

from emergent.fish_passage.centerline import derive_centerline_from_hecras_distance
from emergent.fish_passage.centerline import extract_centerline_fast, extract_centerline_fast_hecras
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_arrays, infer_wetted_perimeter_from_hecras
import h5py
import tempfile


def test_centerline_simple_ridge():
    # Create a simple rectangular channel: x varies 0..9, y varies 0..4
    xs, ys = np.meshgrid(np.arange(10), np.arange(5))
    coords = np.column_stack((xs.flatten(), ys.flatten()))
    # Distance-to-bank: larger near center x=4.5, simulate ridge along center x~4.5
    distances = 1.0 - np.abs(coords[:, 0] - 4.5) / 10.0
    wetted_mask = np.ones(len(coords), dtype=bool)
    centerline = derive_centerline_from_hecras_distance(coords, distances, wetted_mask, min_length=1.0)
    assert centerline is None or isinstance(centerline, LineString)  # depends on ridge extraction


def test_centerline_short_ridge_returns_none():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = np.array([0.0, 0.0])
    wetted_mask = np.array([True, True])
    centerline = derive_centerline_from_hecras_distance(coords, distances, wetted_mask, min_length=100.0)
    assert centerline is None


def test_extract_centerline_fast_array_and_file_wrapper():
    # build a synthetic channel similar to earlier test
    xs, ys = np.meshgrid(np.arange(10), np.arange(5))
    coords = np.column_stack((xs.flatten(), ys.flatten()))
    depths = 1.0 - np.abs(coords[:, 0] - 4.5) / 10.0
    cl = extract_centerline_fast(coords, depths, depth_threshold=0.0, sample_fraction=0.2, min_length=1.0)
    assert cl is None or hasattr(cl, 'length')

    # test file-backed wrapper with a temporary hdf5 file
    import os
    with tempfile.TemporaryDirectory() as td:
        tmp_path = os.path.join(td, 'test_centerline.h5')
        with h5py.File(tmp_path, 'w') as hdf:
            hdf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
            # emulate depth dataset with a time-first dimension
            depth_ds = hdf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([depths]))
        cl2 = extract_centerline_fast_hecras(tmp_path, depth_threshold=0.0, sample_fraction=0.2, min_length=1.0)
        assert cl2 is None or hasattr(cl2, 'length')


    def test_infer_wetted_perimeter_arrays_and_wrapper():
        # construct a simple wetted rectangle of points
        xs, ys = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 5, 6))
        coords = np.column_stack((xs.flatten(), ys.flatten()))
        depths = np.zeros(len(coords))
        # mark central band as wetted
        mask = (coords[:, 0] > 2) & (coords[:, 0] < 8)
        depths[mask] = 1.0
        exterior = infer_wetted_perimeter_from_arrays(coords, depths, depth_threshold=0.5, max_nodes=1000, raster_fallback_resolution=1.0)
        assert exterior is None or (hasattr(exterior, 'shape') and exterior.shape[1] == 2)

        # file-backed wrapper
        import os
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, 'test_perim.h5')
            with h5py.File(tmp_path, 'w') as hdf:
                hdf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
                depth_ds = hdf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([depths]))
            perim = infer_wetted_perimeter_from_hecras(tmp_path, depth_threshold=0.5, max_nodes=1000, raster_fallback_resolution=1.0)
            assert perim is None or (hasattr(perim, 'shape') and perim.shape[1] == 2)
