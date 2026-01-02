"""Simple script to run a salmon ABM simulation for testing."""
import numpy as np
from emergent.salmon_abm.simulation import simulation

# Create simulation
sim = simulation(
    model_dir='outputs/quick_test',
    model_name='quick',
    crs='EPSG:26905',
    basin='nuyakuk',
    water_temp=7.0,
    start_polygon='data/salmon_abm/at_falls.shp',
    env_files={},
    longitudinal_profile='data/salmon_abm/longitudinal.shp',
    num_timesteps=50,
    num_agents=20,
    db_path='outputs/quick_test/quick.h5',
    output_write_mode='full'
)

print(f"Created simulation with {sim.num_agents} agents, {sim.num_timesteps} timesteps")
print(f"Output: {sim.model_dir}")

# Run simulation
dt = 1.0
for t in range(sim.num_timesteps):
    if t % 10 == 0:
        print(f"Step {t}/{sim.num_timesteps}")
    sim.current_step = t
    sim.timestep(t, dt)

print("Simulation complete!")
sim.close()
