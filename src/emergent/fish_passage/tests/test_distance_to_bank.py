import numpy as np
from emergent.fish_passage.geometry import compute_distance_to_bank


def make_grid_coords(nx, ny, spacing=1.0, origin=(0.0, 0.0)):
    ox, oy = origin
    xs = np.repeat(np.linspace(ox, ox + (nx - 1) * spacing, nx), ny)
    ys = np.tile(np.linspace(oy, oy + (ny - 1) * spacing, ny), nx)
    return np.column_stack([xs, ys])


def test_distance_to_bank_simple_square():
    nx, ny = 10, 10
    coords = make_grid_coords(nx, ny, spacing=1.0, origin=(0.0, 0.0))
    # wetted mask: everything true
    wetted_mask = np.ones(nx * ny, dtype=bool)
    # perimeter indices: boundary cells (indices by flattened ordering)
    perim = []
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                perim.append(i * ny + j)
    perim = np.array(perim, dtype=int)
    d = compute_distance_to_bank(coords, wetted_mask, perim, median_spacing=1.0)
    # perimeter entries must be zero
    assert np.allclose(d[perim], 0.0)
    # interior point (center) should have positive finite distance
    center_idx = (nx//2) * ny + (ny//2)
    assert d[center_idx] > 0.0
    assert np.isfinite(d[center_idx])
