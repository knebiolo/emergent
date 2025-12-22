import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np


def test_actual_current_sampling_basic():
    port = "Baltimore"
    curr_fn = get_current_fn(port)
    lons = np.array([-76.5, -76.4, -76.3])
    lats = np.array([39.2, 39.3, 39.4])
    LON, LAT = np.meshgrid(lons, lats)
    now = datetime.utcnow()
    uv = curr_fn(LON.ravel(), LAT.ravel(), now)
    assert uv.shape[1] == 2
