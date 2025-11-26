import pytest
pytestmark = pytest.mark.slow

import sys, time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn


def check_harbor(port, lon, lat):
    current_fn = get_current_fn(port=port, start=datetime.now())
    res = current_fn(np.array([lon]), np.array([lat]), datetime.now())
    return res


def test_multiple_harbors_basic():
    assert check_harbor('Seattle', -122.34, 47.60) is not None
    assert check_harbor('Rosario Strait', -122.70, 48.62) is not None
    assert check_harbor('Galveston', -94.80, 29.30) is not None
