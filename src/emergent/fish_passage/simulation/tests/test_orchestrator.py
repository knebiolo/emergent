import numpy as np

def test_orchestrator_runs_and_logs():
    from emergent.fish_passage.simulation.orchestrator import Orchestrator
    o = Orchestrator(n_agents=3, warmup_numba=False)
    called = {'hook': False}

    def hook(t, out, pid_out):
        called['hook'] = True

    o.add_hook(hook)
    log = o.run(steps=2, dt=1.0)
    assert isinstance(log, list)
    assert len(log) == 2
    assert called['hook'] is True
    # entries should contain finite arrays
    for entry in log:
        assert np.all(np.isfinite(entry['positions']))
        assert np.all(np.isfinite(entry['speeds']))
        assert np.all(np.isfinite(entry['battery']))
