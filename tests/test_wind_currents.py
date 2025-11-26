import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
import numpy as np
from datetime import datetime


def test_wind_and_current_grid_sample():
    port = "Galveston"
    curr_fn = get_current_fn(port)
    wind_fn = get_wind_fn(port)
    lon_min, lon_max = -96.0, -94.0
    lat_min, lat_max = 28.5, 30.0
    n = 3
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)
    LON, LAT = np.meshgrid(lons, lats)
    now = datetime.utcnow()
    uvc = curr_fn(LON.ravel(), LAT.ravel(), now)
    uvw = wind_fn(LON.ravel(), LAT.ravel(), now)
    assert uvc.shape[1] == 2
    assert uvw.shape[1] == 2
