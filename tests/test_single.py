import pytest
pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.ofs_loader import get_current_fn
from datetime import datetime
import numpy as np


def test_single_sampler_loads():
    current_fn = get_current_fn(port='Seattle', start=datetime.now())
    result = current_fn(np.array([-122.34]), np.array([47.60]), datetime.now())
    assert result.shape[1] == 2
