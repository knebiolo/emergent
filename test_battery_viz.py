"""Quick test to verify battery visualization is working."""
import h5py
import numpy as np

# Check the fatigue diagnostic file
h5_path = "outputs/fatigue_diagnostic/fatigue_test_diagnostics.h5"

with h5py.File(h5_path, 'r') as f:
    print("HDF5 file contents:")
    print(f.keys())
    
    if 'agent_data' in f:
        print("\nAgent data keys:")
        print(f['agent_data'].keys())
        
        if 'battery' in f['agent_data']:
            battery = f['agent_data/battery'][:]
            print(f"\nBattery shape: {battery.shape}")
            print(f"Battery dtype: {battery.dtype}")
            print(f"Battery range: {np.min(battery):.4f} to {np.max(battery):.4f}")
            print(f"Battery mean: {np.mean(battery):.4f}")
            
            # Check first and last timestep
            print(f"\nFirst timestep battery stats:")
            print(f"  min={np.min(battery[0]):.4f}, max={np.max(battery[0]):.4f}, mean={np.mean(battery[0]):.4f}")
            print(f"\nLast timestep battery stats:")
            print(f"  min={np.min(battery[-1]):.4f}, max={np.max(battery[-1]):.4f}, mean={np.mean(battery[-1]):.4f}")
            
        if 'X' in f['agent_data']:
            X = f['agent_data/X'][:]
            print(f"\nPosition X shape: {X.shape}")
            print(f"Number of timesteps: {X.shape[0]}")
            print(f"Number of agents: {X.shape[1]}")
