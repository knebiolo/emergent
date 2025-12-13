import numpy as np


class PID_controller:
    """Minimal PID controller used by simulations during migration.

    Provides `update(error, dt, status)` returning P,I,D terms summed.
    Keeps internal integral and previous error state per-agent.
    """
    def __init__(self, n_agents, k_p=0.0, k_i=0.0, k_d=0.0, tau_d=1.0):
        n = int(np.round(n_agents))
        self.k_p = np.array([k_p])
        self.k_i = np.array([k_i])
        self.k_d = np.array([k_d])
        self.tau_d = tau_d
        self.integral = np.zeros((n, 2), dtype=float)
        self.previous_error = np.zeros((n, 2), dtype=float)
        self.derivative_filtered = np.zeros((n, 2), dtype=float)
        # Simple PID plane parameters (not used heavily here)
        self.P_params = np.array([0.0, 0.0, 1.0])
        self.I_params = np.array([0.0, 0.0, 0.0])
        self.D_params = np.array([0.0, 0.0, 0.0])

    def interp_PID(self):
        """Placeholder to compute PID parameter planes if needed.

        For now this is a no-op that ensures callers can call it safely.
        """
        return

    def update(self, error, dt, status):
        """Update controller with `error` shaped (n_agents,2), returns control output."""
        err = np.asarray(error, dtype=float)
        status = np.asarray(status)
        mask = (status == 3)
        # integrate only where not masked
        self.integral = np.where(~mask[:, None], self.integral + err, self.integral)
        derivative = err - self.previous_error
        self.previous_error = err.copy()
        p_term = self.k_p[:, None] * err
        i_term = self.k_i[:, None] * self.integral
        d_term = self.k_d[:, None] * derivative
        out = p_term + i_term + d_term
        return out
