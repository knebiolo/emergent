import numpy as np
from emergent.salmon_abm.viewer_v3 import mesh_builder


def test_build_mesh_basic():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    vals = np.array([0.0, 0.1, 0.2])
    verts, faces, colors = mesh_builder.build_mesh(pts, vals, vert_exag=1.0)
    assert verts.shape[0] == 3
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3
    assert colors.shape[0] == 3
    assert colors.shape[1] == 4
