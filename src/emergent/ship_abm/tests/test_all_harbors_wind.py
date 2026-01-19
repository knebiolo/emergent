import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime, timedelta
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS


def _center(bounds):
    lon = (bounds['minx'] + bounds['maxx']) / 2.0
    lat = (bounds['miny'] + bounds['maxy']) / 2.0
    return lon, lat


def test_all_harbors_wind_basic():
    # Use a recent timestamp to avoid huge monthly archive downloads
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    for port_name, bounds in SIMULATION_BOUNDS.items():
        bbox = (bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'])
        wind_fn = wind_sampler(bbox, start)
        lon, lat = _center(bounds)
        res = wind_fn(np.array([lon]), np.array([lat]), start)
        arr = np.asarray(res)
        assert arr.shape == (1, 2)
        assert np.any(np.isfinite(arr))
