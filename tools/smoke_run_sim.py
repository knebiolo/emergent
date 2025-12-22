from emergent.salmon_abm.simulation import simulation
import os

# minimal env_files empty list (io.enviro_import used only if files provided)
env_files = []

sim = simulation(
    model_dir=os.path.abspath('.'),
    model_name='smoke_sim',
    crs=4326,
    basin='TestBasin',
    water_temp=10.0,
    start_polygon=None,
    env_files=env_files,
    longitudinal_profile=None,
    fish_length=500,
    num_timesteps=10,
    num_agents=5,
    use_gpu=False,
    pid_tuning=False,
)

status = sim.run(n=3, dt=1.0, return_status=True)
print('Run status:', status)
sim.close()
