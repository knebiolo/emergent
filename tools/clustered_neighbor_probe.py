import numpy as np, os
from emergent.salmon_abm.simulation import simulation

# small deterministic clustered probe
outdir = 'outputs/diagnostics'
model_name = 'clustered_probe'
# create simulation with 10 agents and 10 timesteps
sim = simulation(model_dir='outputs/diagnostics', model_name=model_name, crs=None, basin=None, water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None, num_timesteps=10, num_agents=10)
# place agents in a tight cluster around a point
cx, cy = 549800.0, 6641300.0
offsets = np.linspace(-0.5, 0.5, 10)
sim.X = cx + offsets
sim.Y = cy + offsets * 0.2
sim.prev_X = sim.X.copy()
sim.prev_Y = sim.Y.copy()
# set neighbor buffer radius small so neighbors are recognized
sim.neighbor_buffer_radius = 5.0
# enable debug dumps
sim.debug_behavior = True
sim.debug_movement = True
# run a few steps
for step in range(5):
    sim.current_step = step
    sim.timestep(step, 1.0)
print('Done. NPZs in', outdir)
