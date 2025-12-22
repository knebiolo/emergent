import math
import numpy as np
from emergent.ship_abm import angle_utils as au


def test_wrap_rad_basic():
    assert abs(au.wrap_rad(math.pi - 1e-6) - (math.pi - 1e-6)) < 1e-9
    assert abs(au.wrap_rad(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-9


def test_wrap_deg_basic():
    assert au.wrap_deg(179.0) == 179.0
    assert abs(au.wrap_deg(181.0) - (-179.0)) < 1e-9


def test_heading_diff_rad_examples():
    # hd = -179.5°, psi = +179° -> expected ~ +1.5° (= 1.5 * pi/180 rad)
    hd = math.radians(-179.5)
    psi = math.radians(179.0)
    diff = au.heading_diff_rad(hd, psi)
    assert abs(math.degrees(diff) - 1.5) < 1e-6


def test_heading_diff_broadcasting():
    hd = np.array([0.0, math.pi - 0.1])
    psi = 2.0 * math.pi + np.array([0.0, -math.pi + 0.05])
    d = au.heading_diff_rad(hd, psi)
    # ensure shape and reasonable values
    assert d.shape == (2,)
    assert abs(d[0] - 0.0) < 1e-9
    assert abs(math.degrees(d[1]) - 5.0) > 0  # just sanity check
