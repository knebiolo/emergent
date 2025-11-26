"""Run a small smoke test for heading_error semantics without importing the full simulation module.
This avoids heavy UI/geospatial imports that break pytest in this environment.
"""
import numpy as np
from emergent.ship_abm.angle_utils import heading_diff_deg


def heading_error_shim(actual_deg: np.ndarray, commanded_deg: np.ndarray) -> np.ndarray:
    return heading_diff_deg(commanded_deg, actual_deg)


def run_tests():
    # Case 1: commanded -170, actual +170 -> expected 20
    actual = np.array([170.0])
    commanded = np.array([-170.0])
    err = heading_error_shim(actual, commanded)
    print('case1 err =', err)
    assert np.allclose(err, np.array([20.0]))

    # Case 2: commanded 10, actual 5 -> err = 5
    actual = np.array([5.0])
    commanded = np.array([10.0])
    err = heading_error_shim(actual, commanded)
    print('case2 err =', err)
    assert np.allclose(err, np.array([5.0]))

    # Vectorized
    actual = np.array([0.0, 179.0, -179.0])
    commanded = np.array([10.0, -179.5, 179.5])
    err = heading_error_shim(actual, commanded)
    print('vector err =', err)
    assert err.shape == actual.shape
    assert np.all(np.abs(err) <= 180.0)

    print('All smoke tests passed')

if __name__ == '__main__':
    run_tests()
