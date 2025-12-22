import pytest

pytestmark = pytest.mark.slow

import sys
import time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np


def test_quick_current_sampler():
    start = time.time()
    current_fn = get_current_fn(port="Baltimore", start=datetime.now())
    lons = np.array([-76.45])
    lats = np.array([39.30])
    result = current_fn(lons, lats, datetime.now())
    assert result.shape[0] >= 1
