import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn


def test_spatial_variation_baltimore():
    current_fn = get_current_fn(port="Baltimore", start=datetime.now())
    locations = [(-76.60, 39.20), (-76.45, 39.27), (-76.45, 39.35), (-76.45, 39.42), (-76.30, 39.48)]
    now = datetime.now()
    vals = []
    for lon, lat in locations:
        res = current_fn(np.array([lon]), np.array([lat]), now)
        vals.append((res[0,0], res[0,1]))
    assert len(vals) == 5
