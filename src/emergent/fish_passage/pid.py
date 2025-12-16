"""PID controller helpers extracted from legacy sockeye.

This module provides a lightweight `PID_controller` suitable for simulations.
It intentionally keeps optional dependencies optional (pandas, scipy) and
falls back to safe defaults when tuning data is not available.
"""
from typing import Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None


class PID_controller:
    def __init__(self, n_agents: int, k_p: float = 0.0, k_i: float = 0.0, k_d: float = 0.0, tau_d: float = 1.0):
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        n = int(round(n_agents))
        self.integral = np.zeros((n, 2), dtype=float)
        self.previous_error = np.zeros((n, 2), dtype=float)
        self.derivative_filtered = np.zeros((n, 2), dtype=float)

        # PID plane parameters (a, b, c) for P/I/D: defaults applied by interp_PID
        self.P_params = (0.0, 0.0, 1.0)
        self.I_params = (0.0, 0.0, 0.0)
        self.D_params = (0.0, 0.0, 0.0)

        try:
            self.interp_PID()
        except Exception:
            # keep safe defaults
            pass

    def update(self, error: np.ndarray, dt: float, status: np.ndarray) -> np.ndarray:
        """Compute PID output for an array of agent errors.

        Parameters
        - error: array shaped (n_agents, 2)
        - dt: timestep (unused in simple form)
        - status: array shaped (n_agents,) where status==3 indicates fatigued/dead

        Returns
        - control output array shaped (n_agents, 2)
        """
        mask = (status == 3)
        # ensure arrays
        error = np.asarray(error, dtype=float)
        status = np.asarray(status)

        # update integrals only for non-masked agents
        mask_exp = mask[:, None]
        self.integral = np.where(~mask_exp, self.integral + error, self.integral)

        derivative = error - self.previous_error
        self.previous_error = error.copy()

        p_term = self.k_p * error
        i_term = self.k_i * self.integral
        d_term = self.k_d * derivative

        out = p_term + i_term + d_term
        # zero-out masked agents
        out = np.where(~mask_exp, out, 0.0)
        return out

    def interp_PID(self) -> None:
        """Attempt to load PID tuning CSV and fit plane models for P/I/D.

        If pandas or scipy are unavailable, or the CSV is missing, falls back
        to constant gains (defaults set in __init__).
        """
        if pd is None or curve_fit is None:
            return

        try:
            import os
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pid_optimize_Nushagak.csv')
            df = pd.read_csv(data_dir)
            length = df.loc[:, 'fish_length'].values
            velocity = df.loc[:, 'avg_water_velocity'].values
            P = df.loc[:, 'p'].values
            I = df.loc[:, 'i'].values
            D = df.loc[:, 'd'].values

            def plane_model(coords, a, b, c):
                length_arr, velocity_arr = coords
                return a * length_arr + b * velocity_arr + c

            self.P_params, _ = curve_fit(plane_model, (length, velocity), P)
            self.I_params, _ = curve_fit(plane_model, (length, velocity), I)
            self.D_params, _ = curve_fit(plane_model, (length, velocity), D)
        except Exception:
            # keep defaults
            return

    def PID_func(self, velocity, length) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute P, I, D gains from planar models given `velocity` and `length`.

        Returns 1-D numpy arrays for P, I, D matching broadcasted input shapes.
        """
        a_P, b_P, c_P = self.P_params
        a_I, b_I, c_I = self.I_params
        a_D, b_D, c_D = self.D_params

        vel = np.asarray(velocity)
        leng = np.asarray(length)
        if vel.ndim == 0:
            vel = np.full(1, float(vel))
        if leng.ndim == 0:
            leng = np.full(1, float(leng))

        try:
            vel_b, leng_b = np.broadcast_arrays(vel, leng)
        except ValueError:
            vel_b = vel.ravel()
            leng_b = np.broadcast_to(leng.ravel(), vel_b.shape)

        P = a_P * leng_b + b_P * vel_b + c_P
        I = a_I * leng_b + b_I * vel_b + c_I
        D = a_D * leng_b + b_D * vel_b + c_D

        return np.asarray(P).ravel(), np.asarray(I).ravel(), np.asarray(D).ravel()


def simulate_pid_response(pid: PID_controller, errors_sequence: np.ndarray, dt: float, status_sequence: np.ndarray) -> np.ndarray:
    """Simulate PID controller over a sequence of errors.

    - errors_sequence: shape (timesteps, n_agents, 2)
    - status_sequence: shape (timesteps, n_agents)

    Returns outputs array shaped (timesteps, n_agents, 2)
    """
    errors_sequence = np.asarray(errors_sequence)
    status_sequence = np.asarray(status_sequence)
    timesteps = errors_sequence.shape[0]
    n_agents = errors_sequence.shape[1]
    outputs = np.zeros_like(errors_sequence, dtype=float)
    for t in range(timesteps):
        outputs[t] = pid.update(errors_sequence[t], dt, status_sequence[t])
    return outputs


def attach_pid_controller(sim, k_p: float = 1.0, k_i: float = 0.0, k_d: float = 0.0):
    """Attach a PID_controller to a simulation-like object.

    The function expects `sim` to have attribute `num_agents` (int). It will
    create a `PID_controller` with the provided gains and assign it to
    `sim.pid_controller`.
    """
    n = getattr(sim, 'num_agents', None)
    if n is None:
        raise AttributeError('Simulation object must have `num_agents` attribute')
    sim.pid_controller = PID_controller(n, k_p=k_p, k_i=k_i, k_d=k_d)
    return sim.pid_controller
