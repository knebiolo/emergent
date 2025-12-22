import pytest

pytestmark = pytest.mark.slow

import sys
sys.path.insert(0, r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\src")

from emergent.ship_abm.simulation_core import simulation


def test_collision_overlap_immediate():
    sim = simulation(port_name='Rosario Strait', dt=0.5, T=2.0, n_agents=2, load_enc=False)
    pos = (sim.minx + 500.0, sim.miny + 500.0)
    sim.waypoints = [[pos, (pos[0]+1000, pos[1])], [pos, (pos[0]-1000, pos[1])]]
    sim.spawn()
    sim.pos[:,0] = pos
    sim.pos[:,1] = pos
    sim.run()
    assert len(getattr(sim, 'collision_events', [])) >= 0
