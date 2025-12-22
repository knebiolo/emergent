"""PID controller implementation extracted from sockeye.py.

This file provides `PID_controller` with the same public interface used
by the legacy code. It intentionally keeps numpy arrays and avoids
GPU-specific code; the controller operates on numpy arrays passed in.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.optimize import curve_fit


class PID_controller:
    def __init__(self, n_agents, k_p = 0., k_i = 0., k_d = 0., tau_d = 1):
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        self.integral = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.previous_error = np.zeros((np.round(n_agents,0).astype(np.int32),2))
        self.derivative_filtered = np.zeros((np.round(n_agents,0).astype(np.int32),2))

    def update(self, error, dt, status):
        mask = np.where(status == 3,True,False)
        self.integral = np.where(~mask, self.integral + error, self.integral)
        derivative = error - self.previous_error
        self.previous_error = error
        p_term = self.k_p[:, np.newaxis] * error
        i_term = self.k_i[:, np.newaxis] * self.integral
        d_term = self.k_d[:, np.newaxis] * derivative
        return np.where(~mask,p_term + i_term + d_term,0.0)

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
        a_P = self.P_params[0]
        b_P = self.P_params[1]
        c_P = self.P_params[2]
        a_I = self.I_params[0]
        b_I = self.I_params[1]
        c_I = self.I_params[2]
        a_D = self.D_params[0]
        b_D = self.D_params[1]
        c_D = self.D_params[2]
        P = a_P * length + b_P * velocity + c_P
        I = a_I * length + b_I * velocity + c_I
        D = a_D * length + b_D * velocity + c_D
        return P, I, D


__all__ = ["PID_controller"]
