import numpy as np
from emergent.salmon_abm import utils


def test_geo_pixel_roundtrip():
    # simple affine: x = col, y = row
    transform = (1, 0, 0, 0, 1, 0)
    row, col = 10, 5
    x, y = utils.pixel_to_geo(row, col, transform)
    r2, c2 = utils.geo_to_pixel(x, y, transform)
    assert (r2, c2) == (row, col)


def test_standardize_shape_and_slices():
    arr = np.zeros((20, 30))
    assert utils.standardize_shape(arr) == (20, 30)
    srow, scol = utils.determine_slices((5, 5), 2, (20, 30))
    assert srow.start == 3 and srow.stop == 8
    assert scol.start == 3 and scol.stop == 8


def test_linear_interpolate():
    a = np.array([0.0, 1.0])
    b = np.array([2.0, 3.0])
    mid = utils.linear_interpolate(a, b, 0.5)
    assert np.allclose(mid, np.array([1.0, 2.0]))


def test_calculate_front_mask():
    vals = np.array([[0.0, 0.1, 0.2], [0.2, 2.0, 0.3]])
    mask = utils.calculate_front_mask(vals, axis=1)
    assert mask.shape == vals.shape
    # at least one True expected (the jump to 2.0)
    assert mask.any()
import numpy as np
import importlib.util
from pathlib import Path

# load utils module by path to avoid importing package-level __init__
tests_dir = Path(__file__).resolve().parent
pkg_dir = tests_dir.parent
utils_path = pkg_dir / 'utils.py'
spec = importlib.util.spec_from_file_location('salmon_abm.utils', str(utils_path))
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class DummyTransform:
    def __init__(self, a=1.0, c=0.0, e=1.0, f=0.0):
        self.a = a
        self.c = c
        self.e = e
        self.f = f

    def __invert__(self):
        # for testing, inversion returns self and __mul__ implements inverse mapping
        return self

    def __mul__(self, xy):
        x, y = xy
        # invert the forward mapping x = c + a*(col+0.5)
        col = (x - self.c) / self.a - 0.5
        row = (y - self.f) / self.e - 0.5
        return (col, row)


def test_standardize_shape():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    std = utils.standardize_shape(arr, target_shape=(5, 5), fill_value=0)
    assert std.shape == (5, 5)
    assert std[0, 0] == 1
    assert std[2, 2] == 9
    assert std[3, 3] == 0


def test_determine_slices_from_headings():
    headings = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    slices = utils.determine_slices_from_headings(headings, num_slices=4)
    assert list(slices) == [0, 1, 2, 3]


def test_calculate_front_masks():
    # Single agent at origin heading 0 radians (pointing +x)
    headings = np.array([0.0])
    xs = np.linspace(-1, 1, 3)
    ys = np.linspace(-1, 1, 3)
    xv, yv = np.meshgrid(xs, ys)
    x_coords = np.expand_dims(xv, 0)
    y_coords = np.expand_dims(yv, 0)
    agent_x = np.array([0.0])
    agent_y = np.array([0.0])
    masks = utils.calculate_front_masks(headings, x_coords, y_coords, agent_x, agent_y, behind_value=0)
    # Cells with x > 0 should be marked as front (1) (elementwise)
    expected = xv > 0
    assert np.array_equal((masks[0] == 1), expected)


def test_pixel_geo_roundtrip():
    transform = DummyTransform(a=1.0, c=0.0, e=1.0, f=0.0)
    rows, cols = utils.geo_to_pixel([0.0, 1.0], [0.0, 1.0], transform)
    # expect indices near 0 (rounding behavior yields 0 for 0.5)
    assert rows[0] == 0
    assert cols[0] == 0
    assert rows[1] == 0
    assert cols[1] == 0


def test_get_arr():
    arr = utils.get_arr(False)
    assert arr is np
    arr_gpu = utils.get_arr(True)
    assert getattr(arr_gpu, "__name__", "") in ("numpy", "cupy")
