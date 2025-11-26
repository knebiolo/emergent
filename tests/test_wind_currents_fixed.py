import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS
import numpy as np


def test_wind_currents_fixed_basic():
    port = 'Seattle'
    cfg = SIMULATION_BOUNDS.get(port)
    assert cfg is not None
    curr_fn = get_current_fn(port)
    wind_fn = get_wind_fn(port)
    n = 3
    lons = np.linspace(cfg['minx'], cfg['maxx'], n)
    lats = np.linspace(cfg['miny'], cfg['maxy'], n)
    LON, LAT = np.meshgrid(lons, lats)
    now = datetime.utcnow()
    uvc = curr_fn(LON.ravel(), LAT.ravel(), now)
    uvw = wind_fn(LON.ravel(), LAT.ravel(), now)
    assert uvc.shape[1] == 2
    assert uvw.shape[1] == 2
