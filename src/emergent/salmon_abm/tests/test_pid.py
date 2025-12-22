import numpy as np
import importlib.util
from pathlib import Path

# load pid module by path to avoid importing package-level __init__
tests_dir = Path(__file__).resolve().parent
pkg_dir = tests_dir.parent
pid_path = pkg_dir / 'pid.py'
spec = importlib.util.spec_from_file_location('salmon_abm.pid', str(pid_path))
pid_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pid_mod)
PID_controller = pid_mod.PID_controller


def test_pid_update_zero():
    pid = PID_controller(2, k_p=1.0, k_i=0.5, k_d=0.1)
    error = np.zeros((2, 2))
    status = np.zeros(2, dtype=int)
    out = pid.update(error, dt=0.1, status=status)
    assert np.allclose(out, 0.0)


def test_pid_update_nonzero():
    pid = PID_controller(2, k_p=1.0, k_i=0.5, k_d=0.1)
    error = np.array([[1.0, -1.0], [0.5, 0.5]])
    status = np.array([0, 0])
    out1 = pid.update(error, dt=0.1, status=status)
    # output should be finite and same shape
    assert out1.shape == error.shape
    assert np.isfinite(out1).all()


def test_pid_single_agent_proportional():
    pid = PID_controller(1, k_p=2.0, k_i=0.0, k_d=0.0)
    err = np.array([[1.0, 0.0]])
    out = pid.update(err, dt=1.0, status=None)
    assert out.shape == (1, 2)
    assert np.allclose(out, np.array([[2.0, 0.0]]))


def test_pid_integral_accumulation():
    pid = PID_controller(1, k_p=0.0, k_i=1.0, k_d=0.0)
    err = np.array([[1.0, 0.0]])
    pid.update(err, dt=1.0, status=None)
    out2 = pid.update(err, dt=1.0, status=None)
    # integral = 2, so output should reflect that
    assert np.allclose(out2, np.array([[2.0, 0.0]]))


def test_pid_vectorized_agents():
    pid = PID_controller(3, k_p=[1.0, 2.0, 3.0], k_i=0.0, k_d=0.0)
    err = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    out = pid.update(err, dt=1.0, status=None)
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], np.array([1.0, 2.0, 3.0]))
