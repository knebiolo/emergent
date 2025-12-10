import numpy as np
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures import create_vector_perimeter_fixture


def test_infer_wetted_perimeter_vector(tmp_path):
    p = tmp_path / 'vector.h5'
    create_vector_perimeter_fixture(str(p))
    out = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.05, verbose=False)
    # implementation may return an ndarray of exterior coords or a dict with 'perimeter_points'
    if isinstance(out, dict):
        pts = out.get('perimeter_points')
    else:
        pts = out
    assert pts is not None
    pts = np.asarray(pts)
    assert pts.ndim == 2 and pts.shape[0] >= 4
import tempfile
import os
import numpy as np
from emergent.fish_passage.centerline import infer_wetted_perimeter_from_hecras
from emergent.fish_passage.tests.fixtures import create_vector_perimeter_fixture


def test_infer_wetted_perimeter_vector(tmp_path):
    p = tmp_path / 'vector.h5'
    create_vector_perimeter_fixture(str(p))
    out = infer_wetted_perimeter_from_hecras(str(p), depth_threshold=0.05, verbose=False)
    # implementation may return an ndarray of exterior coords or a dict with 'perimeter_points'
    if isinstance(out, dict):
        pts = out.get('perimeter_points')
    else:
        pts = out
    assert pts is not None
    pts = np.asarray(pts)
    assert pts.ndim == 2 and pts.shape[0] >= 4

