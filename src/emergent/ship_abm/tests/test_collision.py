import pytest

pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.simulation_core import simulation


def test_collision_headless_small():
    sim = simulation(port_name='Rosario Strait', dt=0.5, T=5.0, n_agents=2, load_enc=False)
    sim.waypoints = [
        [(sim.minx + 200.0, sim.miny + 200.0), (sim.minx + 2000.0, sim.miny + 200.0)],
        [(sim.minx + 2000.0, sim.miny + 400.0), (sim.minx + 200.0, sim.miny + 200.0)],
    ]
    sim.spawn()
    for i, wp in enumerate(sim.waypoints):
        sim.pos[:, i] = wp[0]
    sim.run()
    assert hasattr(sim, 'collision_events')
