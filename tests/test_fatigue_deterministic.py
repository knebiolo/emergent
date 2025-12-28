import numpy as np
from emergent.salmon_abm.simulation import simulation
from emergent.salmon_abm.fatigue import fatigue
from emergent.salmon_abm import hdf5_io


def make_sim(num_agents=3):
    sim = simulation(model_dir='outputs/diagnostics', model_name='det_fatigue', crs=None, basin='nuyakuk',
                     water_temp=10.0, start_polygon=None, env_files=None, longitudinal_profile=None,
                     num_timesteps=10, num_agents=num_agents, db_path='outputs/diagnostics/det_fatigue.h5')
    return sim


def test_battery_depletion_and_recovery():
    sim = make_sim(2)
    # deterministic initial conditions
    sim.battery[:] = np.array([1.0, 0.2])
    sim.sog[:] = np.array([0.5, 0.5])
    sim.heading[:] = np.array([0.0, 0.0])
    # set lengths so bl scaling predictable
    sim.length[:] = np.array([100.0, 100.0])
    # ensure max sustained/sprint thresholds are small to trigger depletion
    sim.max_s_U[:] = np.repeat(0.1, sim.num_agents)
    sim.max_p_U[:] = np.repeat(0.2, sim.num_agents)

    # run for several timesteps and record battery
    bat_hist = []
    for t in range(5):
        sim.current_step = t
        sim.timestep(t, 1.0)
        bat_hist.append(sim.battery.copy())
    bat_hist = np.vstack(bat_hist)
    # battery should not increase for agent starting at 1.0 and should decrease for low battery agent
    assert bat_hist.shape[0] == 5
    assert bat_hist[0,0] >= bat_hist[-1,0]
    assert bat_hist[0,1] >= bat_hist[-1,1]
    # ensure battery values are within [0,1]
    assert np.all(bat_hist >= 0) and np.all(bat_hist <= 1)
    sim.close()


def test_recovery_resets_bout_and_dist():
    sim = make_sim(1)
    # set battery low and swim_behav to station-holding (3) to trigger recovery
    sim.battery[:] = np.array([0.05])
    sim.swim_behav[:] = np.array([3])
    sim.recover_stopwatch[:] = np.array([0.0])
    sim.dist_per_bout[:] = np.array([10.0])
    sim.bout_dur[:] = np.array([5.0])
    sim.current_step = 0
    sim.timestep(0, 1.0)
    # after recovery(), for swim_behav==3, dist_per_bout and bout_dur should be zeroed and battery may increase
    assert sim.dist_per_bout[0] == 0.0
    assert sim.bout_dur[0] == 0.0
    assert sim.recover_stopwatch[0] >= 0.0
    sim.close()
