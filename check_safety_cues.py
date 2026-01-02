"""
Quick diagnostic to check safety cue magnitudes - are they triggering the threshold?
"""

import h5py
import numpy as np

with h5py.File('outputs/pid_diagnostic_test/sim_pid_test.h5', 'r') as h5:
    print("Available keys:", list(h5.keys()))
    if 'agent_data' in h5:
        print("Agent data keys:", list(h5['agent_data'].keys()))
        
        heading_delta = h5['agent_data/heading_delta'][:]  # (N, T)
        
        # Find timesteps with large heading changes
        heading_delta_deg = np.rad2deg(np.abs(heading_delta))
        
        # Get indices of large changes
        large_changes = heading_delta_deg > 5.0  # >5 degrees
        
        print(f"\nLarge heading changes (>5°): {np.sum(large_changes)} out of {large_changes.size} total")
        print(f"Percentage: {100 * np.sum(large_changes) / large_changes.size:.1f}%")
        
        # Find max changes per agent
        for i in range(min(10, heading_delta.shape[0])):
            max_change = np.max(np.abs(heading_delta_deg[i, :]))
            mean_change = np.mean(np.abs(heading_delta_deg[i, :]))
            print(f"Agent {i}: max={max_change:.1f}°, mean={mean_change:.3f}°")
