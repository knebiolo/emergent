import importlib
import sys
import types

from emergent.fish_passage.utils import get_arr


def test_get_arr_returns_numpy_module_by_default():
    arr_mod = get_arr(use_gpu=False)
    assert hasattr(arr_mod, 'array')
    a = arr_mod.array([1, 2, 3])
    assert a.dtype in (int, a.dtype.__class__)


def test_get_arr_falls_back_to_numpy_when_cupy_missing(monkeypatch):
    # Ensure cupy import fails
    monkeypatch.setitem(sys.modules, 'cupy', None)
    arr_mod = get_arr(use_gpu=True)
    assert arr_mod.__name__ == 'numpy'
