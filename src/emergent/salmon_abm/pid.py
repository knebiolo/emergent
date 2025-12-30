"""PID controller implementation extracted from sockeye.py.

This file provides `PID_controller` with the same public interface used
by the legacy code. It intentionally keeps numpy arrays and avoids
GPU-specific code; the controller operates on numpy arrays passed in.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Any, Tuple


class PID_controller:
    def __init__(self, n_agents, k_p = 1.0, k_i = 0.0, k_d = 0.0, tau_d = 1):
        # coerce n_agents to int
        n = int(np.round(n_agents))
        # accept scalar or array gains; store as arrays shape (n,)
        self.k_p = np.asarray(k_p).reshape(-1)
        if self.k_p.size == 1:
            self.k_p = np.repeat(self.k_p, n)
        self.k_i = np.asarray(k_i).reshape(-1)
        if self.k_i.size == 1:
            self.k_i = np.repeat(self.k_i, n)
        self.k_d = np.asarray(k_d).reshape(-1)
        if self.k_d.size == 1:
            self.k_d = np.repeat(self.k_d, n)
        self.tau_d = tau_d
        self.integral = np.zeros((n, 2), dtype=float)
        self.previous_error = np.zeros((n, 2), dtype=float)
        self.derivative_filtered = np.zeros((n, 2), dtype=float)

    def update(self, error, dt, status):
        # status may be None; treat None as all active
        if status is None:
            mask = np.zeros(self.integral.shape[0], dtype=bool)
        else:
            mask = np.where(np.asarray(status) == 3, True, False)
        # ensure error shape is (n,2)
        err = np.asarray(error)
        if err.ndim == 1:
            err = err[:, np.newaxis]
        # update integral for active agents
        self.integral = np.where(~mask[:, np.newaxis], self.integral + err, self.integral)
        derivative = err - self.previous_error
        self.previous_error = err
        p_term = self.k_p[:, np.newaxis] * err
        i_term = self.k_i[:, np.newaxis] * self.integral
        d_term = self.k_d[:, np.newaxis] * derivative
        out = p_term + i_term + d_term
        out = np.where(~mask[:, np.newaxis], out, 0.0)
        return out

    def interp_PID(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/pid_optimize_Nushagak.csv')
        df = pd.read_csv(data_dir)
        length = df.loc[:, 'fish_length'].values
        velocity = df.loc[:, 'avg_water_velocity'].values
        P = df.loc[:, 'p'].values
        I = df.loc[:, 'i'].values
        D = df.loc[:, 'd'].values

        def plane_model(coords, a, b, c):
            x, y = coords
            return a * x + b * y + c

        self.P_params, _ = curve_fit(plane_model, (length, velocity), P)
        self.I_params, _ = curve_fit(plane_model, (length, velocity), I)
        self.D_params, _ = curve_fit(plane_model, (length, velocity), D)

    def PID_func(self, velocity, length):
        """Return (P, I, D) gains for the given (velocity, length).

        If the controller has been fit via `interp_PID()`, this evaluates the
        learned plane models. Otherwise it returns the controller's current
        gains as scalars.
        """
        params = self._trained_pid_params()
        if params is not None:
            (a_P, b_P, c_P), (a_I, b_I, c_I), (a_D, b_D, c_D) = params
            P = a_P * length + b_P * velocity + c_P
            I = a_I * length + b_I * velocity + c_I
            D = a_D * length + b_D * velocity + c_D
            return P, I, D
        return self._controller_pid_gains()

    def _trained_pid_params(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] | None:
        """Return ((aP,bP,cP),(aI,bI,cI),(aD,bD,cD)) or None when untrained."""
        if not (hasattr(self, "P_params") and hasattr(self, "I_params") and hasattr(self, "D_params")):
            return None

        P_params = np.asarray(getattr(self, "P_params"))
        I_params = np.asarray(getattr(self, "I_params"))
        D_params = np.asarray(getattr(self, "D_params"))
        if P_params.size < 3 or I_params.size < 3 or D_params.size < 3:
            return None

        return (
            (float(P_params[0]), float(P_params[1]), float(P_params[2])),
            (float(I_params[0]), float(I_params[1]), float(I_params[2])),
            (float(D_params[0]), float(D_params[1]), float(D_params[2])),
        )

    def _controller_pid_gains(self) -> Tuple[float, float, float]:
        """Return controller gains as scalar floats (legacy-friendly fallback)."""
        return float(np.mean(self.k_p)), float(np.mean(self.k_i)), float(np.mean(self.k_d))


__all__ = ["PID_controller"]
