import numpy as np
from emergent.ship_abm.simulation_core import heading_error


def test_heading_error_sign_and_wrap():
    # commanded  -170 deg, actual +170 deg -> commanded - actual = -340 -> wrapped -> 20 deg
    actual = np.array([170.0])
    commanded = np.array([-170.0])
    err = heading_error(actual, commanded)
    assert np.allclose(err, np.array([20.0]))

    # commanded 10, actual 5 -> err = 5
    actual = np.array([5.0])
    commanded = np.array([10.0])
    err = heading_error(actual, commanded)
    assert np.allclose(err, np.array([5.0]))

    # vectorized case
    actual = np.array([0.0, 179.0, -179.0])
    commanded = np.array([10.0, -179.5, 179.5])
    err = heading_error(actual, commanded)
    # check shapes and reasonable values
    assert err.shape == actual.shape
    assert np.all(np.abs(err) <= 180.0)
