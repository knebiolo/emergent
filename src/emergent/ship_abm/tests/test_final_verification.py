import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np


def test_get_current_fn_stops_on_success():
    current_fn = get_current_fn(port="Baltimore", start=datetime.now())
    res = current_fn(np.array([-76.45]), np.array([39.27]), datetime.now())
    assert res.shape[1] == 2
