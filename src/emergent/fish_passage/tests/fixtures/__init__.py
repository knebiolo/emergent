import numpy as np
import h5py


def create_vector_perimeter_fixture(path):
    """Create a minimal HDF5 file with facepoints and perimeter mapping that yields a simple rectangular ring."""
    # small 3x3 grid of centers
    nx, ny = 3, 3
    xs = np.repeat(np.linspace(0.0, 2.0, nx), ny)
    ys = np.tile(np.linspace(0.0, 2.0, ny), nx)
    coords = np.column_stack([xs, ys]).astype('f4')

    # FacePoints: corners and midpoints around perimeter
    facepoints = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [2.0, 1.0], [2.0, 2.0], [1.0, 2.0],
        [0.0, 2.0], [0.0, 1.0]
    ], dtype='f4')
    # mark all facepoints as perimeter (-1)
    is_perim = np.full((facepoints.shape[0],), -1, dtype='i4')

    # Cells Face and Orientation Info: naive mapping where each cell references a slice
    N = coords.shape[0]
    face_info = np.zeros((N, 2), dtype='i4')
    # assign each cell a start index and count; for simplicity put start=0,count=0
    # real mapping not required for our vector test, we'll rely on perimeter coords mapping

    # Perimeter coordinates (ordered) map to facepoints
    perim_coords = facepoints.copy()

    # Depth dataset: first timestep
    depth = np.array([0.2] * N, dtype='f4')

    with h5py.File(path, 'w') as hf:
        hf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        hf.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Coordinate', data=facepoints)
        hf.create_dataset('Geometry/2D Flow Areas/2D area/FacePoints Is Perimeter', data=is_perim)
        hf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info', data=face_info)
        hf.create_dataset('Geometry/2D Flow Areas/2D area/Perimeter', data=perim_coords)
        hf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([depth]))


def create_raster_perimeter_fixture(path):
    """Create a minimal HDF5 file lacking facepoints so the raster fallback will be used."""
    nx, ny = 10, 6
    xs = np.repeat(np.linspace(0.0, 9.0, nx), ny)
    ys = np.tile(np.linspace(0.0, 5.0, ny), nx)
    coords = np.column_stack([xs, ys]).astype('f4')
    N = coords.shape[0]
    # Create a depth mask with a rectangular wetted region in the middle
    depth = np.zeros((N,), dtype='f4')
    # mark cells with x between 2..6 and y between 1..4 as wetted
    for i in range(N):
        x, y = coords[i]
        if 2.0 <= x <= 6.0 and 1.0 <= y <= 4.0:
            depth[i] = 0.2

    with h5py.File(path, 'w') as hf:
        hf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        hf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([depth]))
