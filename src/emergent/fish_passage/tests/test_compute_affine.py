import numpy as np
from emergent.fish_passage.geometry import compute_affine_from_hecras
from affine import Affine


def make_grid(nx, ny, spacing, origin=(0.0, 0.0)):
    ox, oy = origin
    xs = np.repeat(np.linspace(ox, ox + (nx - 1) * spacing, nx), ny)
    ys = np.tile(np.linspace(oy, oy + (ny - 1) * spacing, ny), nx)
    return np.column_stack([xs, ys])


def test_compute_affine_grid():
    coords = make_grid(20, 15, 2.0, origin=(10.0, 50.0))
    a = compute_affine_from_hecras(coords)
    # cell size should be close to 2.0
    assert abs(a.a - 2.0) < 0.1
    # verify that pixel center of (0,0) maps near origin adjusted by half-cell
    x0, y0 = a * (0, 0)
    assert abs(x0 - (10.0 - 1.0)) < 1e-6


def test_compute_affine_with_target_cell():
    coords = make_grid(10, 10, 5.0, origin=(0.0, 0.0))
    a = compute_affine_from_hecras(coords, target_cell_size=3.0)
    assert abs(a.a - 3.0) < 1e-9
