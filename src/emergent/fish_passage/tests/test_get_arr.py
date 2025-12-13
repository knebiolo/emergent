from emergent.fish_passage.utils import get_arr
import numpy as np


def test_get_arr_defaults_to_numpy():
    arrmod = get_arr()
    assert arrmod is np
    a = arrmod.array([1, 2, 3])
    assert a.sum() == 6
