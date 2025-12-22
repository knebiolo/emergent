"""Run timestep loop with diagnostics for 10 steps on N agents."""
from emergent.salmon_abm.simulation import simulation
import numpy as np

sim = simulation(model_dir='.', model_name='debug', crs=None, basin='test', water_temp=10, start_polygon=None, env_files=[], longitudinal_profile=None, fish_length=200.0, num_timesteps=10, num_agents=100)
for i in range(10):
    sim.timestep(i, 1.0)
    mean_Hz = np.nanmean(sim.Hz)
    mean_thrust = np.nanmean(np.linalg.norm(sim.thrust, axis=1))
    mean_drag = np.nanmean(np.linalg.norm(sim.drag, axis=1))
    mean_speed = np.nanmean(np.linalg.norm(np.stack((sim.x_vel, sim.y_vel), axis=-1), axis=1))
    moved = np.sum((sim.X != sim.prev_X) | (sim.Y != sim.prev_Y))
    print(f'step {i}: mean_Hz={mean_Hz:.3f}, mean_thrust={mean_thrust:.3e}, mean_drag={mean_drag:.3e}, mean_speed={mean_speed:.3f}, moved_agents={moved}')

sim.db.close()
print('db:', sim.db_path)
