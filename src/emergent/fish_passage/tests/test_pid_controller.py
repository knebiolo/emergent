import numpy as np
from emergent.fish_passage.control import PID_controller


def test_pid_basic():
    pid = PID_controller(2, k_p=1.0, k_i=0.5, k_d=0.1)
    error = np.array([[1.0, 0.0], [0.5, -0.5]])
    status = np.array([0, 0])
    out1 = pid.update(error, dt=1.0, status=status)
    assert out1.shape == (2,2)
    # second update with same error increases integral magnitude
    out2 = pid.update(error, dt=1.0, status=status)
    assert np.all(np.abs(out2) >= np.abs(out1))
