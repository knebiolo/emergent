import numpy as np
import h5py


def create_centerline_fixture(path, nx=60, ny=8, spacing=1.0):
    """Create an HDF5 fixture containing a synthetic meandering river centerline's cell centers and depths.

    The river is approximated by sampled sine-wave offsets along x.
    """
    xs = np.linspace(0.0, (nx-1)*spacing, nx)
    ys_line = 2.0 * np.sin(xs / 10.0)
    coords = []
    depths = []
    for i, x in enumerate(xs):
        for j in range(ny):
            y = ys_line[i] + (j - (ny//2)) * 0.5
            coords.append((x, y))
            # depth higher near center (j middle)
            center_offset = abs(j - (ny//2))
            depth = 0.3 if center_offset <= 1 else 0.0
            depths.append(depth)
    coords = np.array(coords, dtype='f4')
    depths = np.array(depths, dtype='f4')

    with h5py.File(path, 'w') as hf:
        hf.create_dataset('Geometry/2D Flow Areas/2D area/Cells Center Coordinate', data=coords)
        hf.create_dataset('Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth', data=np.vstack([depths]))
