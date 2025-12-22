"""Quick smoke runner: instantiate simulation with 100 agents and run 10 timesteps."""
from emergent.salmon_abm.simulation import simulation
import numpy as np

sim = simulation(model_dir='.', model_name='smoke', crs=None, basin='test', water_temp=10, start_polygon=None, env_files=[], longitudinal_profile=None, fish_length=200.0, num_timesteps=10, num_agents=100)

status = sim.run(n=10, dt=1.0)
print('Run returned:', status)
# inspect a few outputs
print('X[0:5]:', sim.X[:5])
print('Y[0:5]:', sim.Y[:5])
print('Hz[0:5]:', sim.Hz[:5])
print('thrust[0]:', sim.thrust[0])
print('drag[0]:', sim.drag[0])

sim.db.close()
print('DB path:', sim.db_path)
