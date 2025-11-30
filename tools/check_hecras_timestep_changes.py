from pathlib import Path
from emergent.salmon_abm.sockeye_SoA import *
import numpy as np

# lightweight runner that builds sim similarly to run_full_sim_bench
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from emergent.salmon_abm.sockeye_SoA import simulation, load_hecras_plan_cached
from pathlib import Path
plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
env_files = {'x_vel': '', 'y_vel': '', 'depth': '', 'wsel': '', 'elev': '', 'vel_dir': '', 'vel_mag': '', 'wetted': ''}

try:
    sim = simulation(model_dir='.', model_name='check_hecras', crs='EPSG:32604', basin='Nushagak River',
                     water_temp=8.0, start_polygon=None, env_files=env_files,
                     longitudinal_profile=None, fish_length=500, num_timesteps=5,
                     num_agents=10,
                     hecras_plan_path=str(plan), hecras_fields=['Cells Minimum Elevation', 'Water Surface',
                                                              'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y'],
                     hecras_k=8, use_hecras=True)

    # preload HECRAS
    load_hecras_plan_cached(sim, sim.hecras_plan_path, field_names=sim.hecras_fields)

    for t in range(5):
        sim.environment()
        import numpy as np
        print(f't={t} mean x_vel:', np.nanmean(sim.x_vel), 'mean y_vel:', np.nanmean(sim.y_vel))
except Exception as e:
    print('Simulation construction failed:', e)
    raise
