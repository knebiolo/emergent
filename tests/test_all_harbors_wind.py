import pytest
pytestmark = pytest.mark.slow

import sys, time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS


def test_all_harbors_wind_basic():
    for port_name, bounds in SIMULATION_BOUNDS.items():
        bbox = (bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'])
        wind_fn = wind_sampler(bbox, datetime.now())
        res = wind_fn(np.array([(bounds['minx']+bounds['maxx'])/2]), np.array([(bounds['miny']+bounds['maxy'])/2]), datetime.now())
        assert res is not None
