import pytest
pytestmark = pytest.mark.slow

import sys, time
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.simulation_core import simulation
from emergent.ship_abm.config import xml_url


def test_enc_progress_background():
    sim = simulation(port_name='Galveston', load_enc=False, verbose=False)
    # Start background load in thread if desired; here we only assert attributes exist
    assert hasattr(sim, 'load_enc_features')
    assert hasattr(sim, '_enc_progress')
