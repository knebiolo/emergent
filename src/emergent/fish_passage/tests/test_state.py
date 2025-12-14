import numpy as np
from emergent.fish_passage.simulation import state as fp_state


def test_sim_sex_and_length_parity():
    seed = 12345
    rng = np.random.default_rng(seed)
    num_agents = 50
    basin = 'Nushagak River'

    # fish_passage outputs
    sex_fp = fp_state.sim_sex(num_agents, basin, rng)

    # legacy sockeye uses rng.choice; we compare distributions rather than bitwise equality
    rng2 = np.random.default_rng(seed)
    sex_legacy = rng2.choice([0, 1], size=num_agents, p=[0.503, 0.497])

    assert sex_fp.shape == sex_legacy.shape
    assert abs(float(np.mean(sex_fp)) - float(np.mean(sex_legacy))) < 0.05

    # Now test sim_length fixed-length path
    lengths_fp = fp_state.sim_length(num_agents, sex_fp, basin, pid_tuning=True, fish_length=500.0, rng=rng)['length']
    assert np.all(lengths_fp == 500.0)
