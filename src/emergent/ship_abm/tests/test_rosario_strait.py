import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn


def test_sscofs_seattle_rosario_basic():
    current_fn = get_current_fn(port="Seattle", start=datetime.now())
    now = datetime.now()
    locs = [(-122.34, 47.60), (-122.67, 48.55), (-122.70, 48.62)]
    for lon, lat in locs:
        res = current_fn(np.array([lon]), np.array([lat]), now)
        assert res.shape[1] == 2
