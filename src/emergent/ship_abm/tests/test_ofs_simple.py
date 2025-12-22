import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from datetime import date, timedelta
from emergent.ship_abm.ofs_loader import regional_keys, BUCKETS
import fsspec


def test_ofs_keys_generation():
    model = 'cbofs'
    today = date.today()
    keys = regional_keys(model, today)
    assert isinstance(keys, list)
    assert len(keys) > 0
