import os
import pytest

RUN_HISTORICAL = os.environ.get("SHIP_ABM_RUN_HISTORICAL") == "1"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_HISTORICAL,
        reason="Historical NOAA archives are large; set SHIP_ABM_RUN_HISTORICAL=1 to enable.",
    ),
]

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

import numpy as np
from datetime import datetime

from emergent.ship_abm.config import SIMULATION_BOUNDS
from emergent.ship_abm.ofs_loader import get_current_fn
from emergent.ship_abm.atmospheric import wind_sampler


def test_dali_historical_currents_time_varying():
    # MV Dali impact ~2024-03-26 05:28 UTC; start 30 min prior
    start = datetime(2024, 3, 26, 4, 58)
    bounds = SIMULATION_BOUNDS["Baltimore"]
    lon = (bounds["minx"] + bounds["maxx"]) / 2.0
    lat = (bounds["miny"] + bounds["maxy"]) / 2.0
    current_fn = get_current_fn(port="Baltimore", start=start, time_window_hours=6.0)
    res = current_fn(np.array([lon]), np.array([lat]), start)
    arr = np.asarray(res)
    assert arr.shape == (1, 2)
    assert np.any(np.isfinite(arr))
    assert hasattr(current_fn, "_source")


def test_dali_historical_wind():
    start = datetime(2024, 3, 26, 4, 58)
    bounds = SIMULATION_BOUNDS["Baltimore"]
    bbox = (bounds["minx"], bounds["maxx"], bounds["miny"], bounds["maxy"])
    lon = (bounds["minx"] + bounds["maxx"]) / 2.0
    lat = (bounds["miny"] + bounds["maxy"]) / 2.0
    wind_fn = wind_sampler(bbox, start)
    res = wind_fn(np.array([lon]), np.array([lat]), start)
    arr = np.asarray(res)
    assert arr.shape == (1, 2)
    assert np.any(np.isfinite(arr))
    assert hasattr(wind_fn, "_source")
