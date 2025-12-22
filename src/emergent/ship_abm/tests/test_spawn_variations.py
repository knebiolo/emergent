import pytest

pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.simulation_core import simulation


def test_spawn_variations_basic():
    cases = {
        'good_lonlat': [[(-122.7, 48.2), (-122.65, 48.2)]],
        'swapped_latlon': [[(48.2, -122.7), (48.2, -122.65)]],
        'strings': [[('-122.7', '48.2'), ('-122.65', '48.2')]],
    }
    for name, wps in cases.items():
        sim = simulation(port_name='Rosario Strait', dt=0.1, T=1, n_agents=1, verbose=False, load_enc=False)
        sim.waypoints = wps
        state0, pos0, psi0, goals = sim.spawn()
        assert pos0.shape[1] == 1
