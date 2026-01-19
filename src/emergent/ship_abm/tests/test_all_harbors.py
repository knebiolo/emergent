import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime, timedelta
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP


def _center(bounds):
    lon = (bounds['minx'] + bounds['maxx']) / 2.0
    lat = (bounds['miny'] + bounds['maxy']) / 2.0
    return lon, lat


def test_all_harbors_basic():
    # Use a recent timestamp to avoid huge monthly archive downloads
    start = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(days=1)
    for port_name, bounds in SIMULATION_BOUNDS.items():
        current_fn = get_current_fn(port=port_name, start=start, time_window_hours=0.0)
        lon, lat = _center(bounds)
        res = current_fn(np.array([lon]), np.array([lat]), start)
        arr = np.asarray(res)
        assert arr.shape == (1, 2)
        assert np.any(np.isfinite(arr))
