import pytest
pytestmark = pytest.mark.slow

# This is an integration-style script that requires OFS and atmospheric data.
# Marked slow to avoid CI runs unless explicitly requested.

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime, timedelta
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.atmospheric import wind_sampler
from emergent.ship_abm.config import SIMULATION_BOUNDS


def test_galveston_env_sampling_basic():
    HARBOR = "Galveston"
    bounds = SIMULATION_BOUNDS[HARBOR]
    START_TIME = datetime.now()
    wind_fn = wind_sampler((bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy']), START_TIME)
    current_fn = get_current_fn(HARBOR, START_TIME)
    lons = np.array([(bounds['minx'] + bounds['maxx'])/2])
    lats = np.array([(bounds['miny'] + bounds['maxy'])/2])
    c = current_fn(lons, lats, START_TIME)
    w = wind_fn(lons, lats, START_TIME)
    assert c is not None
    assert w is not None
