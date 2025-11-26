import math
import numpy as np


def wrap_deg(deg):
    return ((deg + 180.0) % 360.0) - 180.0


def wrap_rad(rad):
    return ((rad + math.pi) % (2 * math.pi)) - math.pi


def test_wrapping_examples():
    # Unwrapped psi (e.g., many rotations) should wrap correctly
    psi_unwrapped_deg = 360.0 * 12 + 179.0  # 12 full turns + 179
    hd_cmd_deg = -179.5

    psi_wr = wrap_deg(psi_unwrapped_deg)
    assert psi_wr == 179.0

    # The logged hd should be wrapped to -179.5
    hd_wr = wrap_deg(hd_cmd_deg)
    assert hd_wr == -179.5

    # difference in controller canonical form err = hd - psi (degrees)
    err_deg = wrap_deg(hd_wr - psi_wr)
    # hd_wr (-179.5) - psi_wr (179.0) = -358.5 -> wrapped -> 1.5 deg
    assert abs(err_deg - 1.5) < 1e-6


def test_wrap_rad_consistent_with_deg():
    # Random samples
    rng = np.random.RandomState(1)
    for _ in range(1000):
        psi = (rng.rand() - 0.5) * 1000.0  # large unwrapped radians
        hd = (rng.rand() - 0.5) * 2 * math.pi
        # convert to deg and wrap
        psi_deg = math.degrees(psi)
        hd_deg = math.degrees(hd)
        wd = wrap_deg(hd_deg - psi_deg)
        wr = math.degrees(wrap_rad(hd - psi))
        assert abs(wd - wr) < 1e-6
