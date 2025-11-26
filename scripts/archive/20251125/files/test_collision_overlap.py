"""Force an immediate overlap collision by spawning two ships at the same position.
"""
from emergent.ship_abm.simulation_core import simulation

port = 'Rosario Strait'
sim = simulation(port_name=port, dt=0.5, T=10.0, n_agents=2, load_enc=False)
# craft identical waypoints so hulls overlap
pos = (sim.minx + 500.0, sim.miny + 500.0)
sim.waypoints = [ [pos, (pos[0]+1000, pos[1])], [pos, (pos[0]-1000, pos[1])] ]
try:
    sim.spawn()
except Exception:
    pass
# force positions to be identical at start
sim.pos[:,0] = pos
sim.pos[:,1] = pos
sim.run()
print('Recorded collisions:', sim.collision_events)
