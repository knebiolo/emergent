import numpy as np

from emergent.fish_passage.pid import PID_controller, simulate_pid_response


def test_pid_update_basic():
    pid = PID_controller(2, k_p=1.0, k_i=0.5, k_d=0.1)
    # two agents, errors for x/y
    error = np.array([[1.0, 0.0], [0.5, -0.5]])
    status = np.array([0, 0])
    # compute expected output using pre-update internal state but account for
    # the fact that `update` adds `error` to the integral before computing i_term.
    pre_integral = pid.integral.copy()
    pre_prev = pid.previous_error.copy()
    expected = pid.k_p * error + pid.k_i * (pre_integral + error) + pid.k_d * (error - pre_prev)
    out = pid.update(error, dt=1.0, status=status)
    assert out.shape == error.shape
    assert np.allclose(out, expected)


def test_pid_masking():
    pid = PID_controller(2, k_p=1.0, k_i=0.0, k_d=0.0)
    error = np.array([[2.0, 2.0], [1.0, 1.0]])
    status = np.array([0, 3])
    out = pid.update(error, dt=1.0, status=status)
    assert out[1, 0] == 0.0 and out[1, 1] == 0.0


def test_simulate_pid_response():
    pid = PID_controller(1, k_p=1.0)
    errors = np.zeros((3, 1, 2))
    errors[0, 0, 0] = 1.0
    status = np.zeros((3, 1), dtype=int)
    outputs = simulate_pid_response(pid, errors, dt=1.0, status_sequence=status)
    assert outputs.shape == errors.shape
    assert outputs[0, 0, 0] != 0.0


def test_attach_pid_controller():
    class DummySim:
        def __init__(self):
            self.num_agents = 5

    sim = DummySim()
    from emergent.fish_passage.pid import attach_pid_controller
    pid = attach_pid_controller(sim, k_p=1.0, k_i=0.0, k_d=0.0)
    assert hasattr(sim, 'pid_controller')
    assert pid is sim.pid_controller
