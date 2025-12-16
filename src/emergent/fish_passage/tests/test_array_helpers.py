import numpy as np

from emergent.fish_passage.utils import standardize_shape


def test_standardize_shape_same_shape():
    a = np.array([[1, 2], [3, 4]])
    out = standardize_shape(a, target_shape=(2, 2), fill_value=0)
    assert np.array_equal(out, a)


def test_standardize_shape_expand():
    a = np.array([[1, 2], [3, 4]])
    out = standardize_shape(a, target_shape=(3, 4), fill_value=-1)
    assert out.shape == (3, 4)
    assert out[0, 0] == 1
    assert out[1, 1] == 4
    assert out[2, 3] == -1


def test_standardize_shape_empty_input():
    a = np.array([]).reshape(0, 0)
    out = standardize_shape(a, target_shape=(2, 2), fill_value=0)
    assert out.shape == (2, 2)
    assert np.all(out == 0)
