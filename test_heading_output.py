"""Quick test to verify heading is being saved to HDF5."""
import numpy as np
import h5py
import sys
sys.path.insert(0, 'src')

from emergent.salmon_abm.simulation import simulation

# Create minimal test simulation
sim = simulation(
    num_agents=10,
    dt=0.1,
    world_bounds=(0, 100, 0, 100),
    use_hecras=False
)

# Run for just 5 timesteps
results = sim.run(
    num_timesteps=5,
    db_path='outputs/test_heading.h5',
    mode='full',
    write_frequency=1
)

# Check if heading was saved
print("\nChecking HDF5 output...")
with h5py.File('outputs/test_heading.h5', 'r') as f:
    print(f"agent_data keys: {list(f['agent_data'].keys())}")
    
    if 'agent_data/heading' in f:
        heading = np.array(f['agent_data/heading'])
        print(f"✓ Heading found! Shape: {heading.shape}")
        print(f"  Sample values (first 3 agents, all timesteps):")
        print(f"  {heading[:3, :]}")
    else:
        print("✗ Heading NOT found in output!")

print("\nTest complete.")
