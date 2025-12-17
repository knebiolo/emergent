import numpy as np

def test_smoke_run_basic():
    from emergent.fish_passage.simulation.smoke_run import run_short_scenario
    pos, spd, batt = run_short_scenario(n_agents=4, steps=2, dt=1.0)
    assert pos.shape == (4,2)
    assert spd.shape == (4,)
    assert batt.shape == (4,)
    # battery levels should be finite numbers
    assert np.all(np.isfinite(batt))
