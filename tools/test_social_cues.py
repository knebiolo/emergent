import numpy as np
from emergent.salmon_abm.simulation import simulation

sim = simulation(model_dir='outputs/diagnostics', model_name='test_social', crs=None, basin=None, water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None, num_timesteps=1, num_agents=4)
# place two close pairs
sim.X = np.array([0.0, 0.2, 10.0, 10.2]) + 549800.0
sim.Y = np.array([0.0, 0.1, 0.0, 0.1]) + 6641300.0
sim.prev_X = sim.X.copy()
sim.prev_Y = sim.Y.copy()
sim.neighbor_buffer_radius = 5.0
sim.debug_behavior = True
sim.debug_movement = True

sim.current_step = 0
sim.timestep(0, 1.0)

import glob
files = glob.glob('outputs/diagnostics/behavior_debug_step_*.npz')
if not files:
    raise SystemExit('No behavior NPZs found')
f = sorted(files)[-1]
data = np.load(f)
print('Inspecting', f)
for k in ['avoid_mag','collision_mag','alignment_mag','cohesion_mag']:
    if k in data:
        print(k, data[k])
    else:
        print(k, 'missing')

# basic assertions
assert np.any(data['cohesion_mag'] > 0), 'cohesion_mag not > 0'
assert np.any(np.isfinite(data['alignment_mag'])), 'alignment_mag not finite'
print('Social cues test passed')
