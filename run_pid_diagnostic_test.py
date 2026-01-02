"""
Quick PID diagnostic test - runs 200 timesteps with a small population.
"""

import sys
import os
sys.path.insert(0, 'src')

from emergent.salmon_abm.simulation import simulation
import numpy as np

# Discover environment files
base_dir = 'data/salmon_abm'
env_files = []
for fn in ["depth.tif", "vel_x.tif", "vel_y.tif", "wsel.tif"]:
    p = os.path.join(base_dir, fn)
    if os.path.exists(p):
        env_files.append(p)

print(f"Found {len(env_files)} environment files")

# Create simulation with minimal configuration
sim = simulation(
    model_dir='outputs/pid_diagnostic_test',
    model_name='pid_test',
    crs=None,
    basin='nuyakuk',
    water_temp=10.0,
    start_polygon='data/salmon_abm/start_loc_river_middle.shp',
    env_files=env_files,
    longitudinal_profile='data/salmon_abm/longitudinal.shp',
    num_timesteps=200,
    num_agents=50,
    db_path='outputs/pid_diagnostic_test/sim_pid_test.h5'
)

print(f"Running PID diagnostic test...")
print(f"  Agents: {sim.num_agents}")
print(f"  Timesteps: 200")
print(f"  Output: {sim.db_path}")

# Run simulation
t = 0.0
dt = 0.1
timestep = 0
max_timesteps = 200

while timestep < max_timesteps:
    sim.timestep(t, dt)
    t += dt
    timestep += 1
    
    if timestep % 50 == 0:
        print(f"  t={t:.1f}s ({timestep} timesteps)")

# Close output
sim.close()

print(f"\nSimulation complete!")
print(f"Output saved to: {sim.db_path}")
print(f"\nRun diagnostic analysis:")
print(f"  python scripts/diagnose_pid_vibration.py {sim.db_path}")


