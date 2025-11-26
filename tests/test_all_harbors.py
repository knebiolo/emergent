import pytest
pytestmark = pytest.mark.slow

import sys, time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.config import SIMULATION_BOUNDS, OFS_MODEL_MAP


def test_all_harbors_basic():
    for port_name, bounds in SIMULATION_BOUNDS.items():
        current_fn = get_current_fn(port=port_name, start=datetime.now())
        lon = (bounds['minx'] + bounds['maxx'])/2
        lat = (bounds['miny'] + bounds['maxy'])/2
        res = current_fn(np.array([lon]), np.array([lat]), datetime.now())
        assert res is not None
