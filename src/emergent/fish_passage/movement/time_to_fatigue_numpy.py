"""NumPy fallback for time-to-fatigue kernel."""
import numpy as np

def time_to_fatigue(energy_states, workload_params=None):
    energy_states = np.asarray(energy_states, dtype=float)
    # trivial linear model: time = energy / workload_rate; default workload_rate=0.1
    rate = 0.1
    if workload_params and 'rate' in workload_params:
        rate = float(workload_params['rate'])
    # avoid division by zero
    rate = np.maximum(rate, 1e-8)
    return energy_states / rate
