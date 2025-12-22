"""Probe behavior arbitration and environment sampling during a short run."""
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm import hdf5_io
import numpy as np

sim = simulation(model_dir='.', model_name='probe', crs=None, basin='test', water_temp=10, start_polygon=None, env_files=[], longitudinal_profile=None, fish_length=200.0, num_timesteps=10, num_agents=100)

print('Initial sample environment depth mean:', np.nanmean(hdf5_io.read_dataset(sim.db, 'environment/depth')))

for i in range(10):
    sim.timestep(i, 1.0)
    # sample a few agents
    sample_idx = [0, 1, 2, 3, 4]
    depths = sim.sample_environment(getattr(sim, 'depth_rast_transform', None), 'depth') if hasattr(sim, 'sample_environment') else None
    # print behavior cues from the behavior module using a fresh instance
    from emergent.salmon_abm.behavior import behavior
    bh = behavior(1.0, sim)
    vel_cue = bh.vel_cue(weight=1.0)
    shallow = bh.shallow_cue(weight=1.0)
    rheo = bh.rheo_cue(weight=1.0)
    print(f'step {i}: mean Hz {np.nanmean(sim.Hz):.3f}, mean depth sample: {np.nanmean(depths) if depths is not None else "n/a"}')
    print(' vel_cue summary min/max:', np.nanmin(vel_cue), np.nanmax(vel_cue))
    print(' shallow summary min/max:', np.nanmin(shallow), np.nanmax(shallow))
    print(' rheo summary min/max:', np.nanmin(rheo), np.nanmax(rheo))

sim.db.close()
print('db:', sim.db_path)
