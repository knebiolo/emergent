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
