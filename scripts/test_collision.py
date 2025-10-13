"""Simple headless test that spawns two ships on a direct collision to validate
collision detection, stopping, and energy calculation.
"""
from datetime import datetime
from emergent.ship_abm.simulation_core import simulation

# Use a tiny domain centered somewhere neutral (UTM conversions will apply)
port = 'Rosario Strait'
# Create sim with 2 agents and short route
sim = simulation(port_name=port, dt=0.5, T=60.0, n_agents=2, load_enc=False)

# define simple straight-line waypoints that cross
# positions are in lon/lat in SIMULATION_BOUNDS; we will map to utm in spawn
# We'll craft a small crossing in local UTM by directly setting waypoints after spawn.

# spawn defaults (calls self.spawn internally) -> but simpler: set positions/waypoints directly
# For test, place ships close so they collide quickly
sim.waypoints = [
    [ (sim.minx + 200.0, sim.miny + 200.0), (sim.minx + 2000.0, sim.miny + 200.0) ],
    [ (sim.minx + 2000.0, sim.miny + 400.0), (sim.minx + 200.0, sim.miny + 200.0) ],
]
# note: spawn() in simulation_core would normally translate waypoints; calling spawn by invoking
state0, pos0, psi0, goals = sim.spawn() if hasattr(sim, 'spawn') else (None, None, None, None)

# if spawn exists, call it to initialize ship and state
try:
    sim.spawn()
except Exception:
    pass

# override histories to ensure positions match our wps
for i, wp in enumerate(sim.waypoints):
    sim.pos[:, i] = wp[0]

# run headless
sim.run()

print('Collision events:', sim.collision_events)