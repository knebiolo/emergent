import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn, get_wind_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS
import datetime as dt


def test_wind_and_current_samplers_basic():
    port = 'Galveston'
    cfg = SIMULATION_BOUNDS.get(port)
    assert cfg is not None
    wind_fn = get_wind_fn(port)
    current_fn = get_current_fn(port)
    lon = (cfg['minx'] + cfg['maxx']) / 2
    lat = (cfg['miny'] + cfg['maxy']) / 2
    now = dt.datetime.utcnow()
    w = wind_fn([lon], [lat], now)
    c = current_fn([lon], [lat], now)
    assert w is not None
    assert c is not None
