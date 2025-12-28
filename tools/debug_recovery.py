from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.fatigue import fatigue
import numpy as np

sim = simulation(model_dir='outputs/diagnostics', model_name='debug_recovery', crs=None, basin='nuyakuk',
                 water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None,
                 num_timesteps=10, num_agents=1, db_path='outputs/diagnostics/debug_recovery.h5')
print('Initial dist_per_bout', sim.dist_per_bout, 'bout_dur', sim.bout_dur, 'swim_behav', sim.swim_behav)
sim.dist_per_bout[:] = np.array([10.0], dtype=float)
sim.bout_dur[:] = np.array([5.0], dtype=float)
sim.swim_behav[:] = np.array([3], dtype=np.int8)
sim.recover_stopwatch[:] = np.array([0.0], dtype=float)
fg = fatigue(0, 1.0, sim)
print('Before recovery: dist_per_bout', sim.dist_per_bout, 'bout_dur', sim.bout_dur, 'battery', sim.battery)
per_rec = fg.recovery()
print('After recovery: dist_per_bout', sim.dist_per_bout, 'bout_dur', sim.bout_dur, 'battery', sim.battery, 'per_rec', per_rec)
