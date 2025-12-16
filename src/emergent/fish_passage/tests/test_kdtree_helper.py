import numpy as np
import pytest

from emergent.fish_passage.utils import safe_build_kdtree


def test_safe_build_kdtree_valid():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = safe_build_kdtree(pts, name='test_tree')
    assert tree is not None
    d, idx = tree.query(np.array([[0.5, 0.5]]), k=1)
    assert idx.shape[0] == 1


def test_safe_build_kdtree_none_and_empty():
    assert safe_build_kdtree(None) is None
    assert safe_build_kdtree(np.array([])) is None
