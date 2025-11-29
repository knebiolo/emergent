"""Run a short headless simulation loop that uses HECRAS mapping for environment fields.

This script creates a simulation instance but monkeypatches `enviro_import` to avoid raster I/O.
It then sets `hecras_plan_path` and fields, runs `environment()` for a few timesteps, and prints timing.
"""
import time
import types
import numpy as np
from pathlib import Path
from emergent.salmon_abm.sockeye_SoA import simulation, load_hecras_plan_cached
import emergent.salmon_abm.sockeye_SoA as sock_mod

plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
start_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\starting_location\starting_location.shp")
long_shp = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\Longitudinal\longitudinal.shp")
print('Using HECRAS plan:', plan)

# Create a minimal env_files dict with placeholders; we'll monkeypatch enviro_import
env_files = {'x_vel':'', 'y_vel':'', 'depth':'', 'wsel':'', 'elev':'', 'vel_dir':'', 'vel_mag':'', 'wetted':''}

# We will use HECRAS-only mode (no raster imports). The simulation constructor
# now accepts `hecras_plan_path` and `use_hecras=True` to preload the KDTree.

# Instantiate simulation with small agent count
# Instantiate simulation (init will skip heavy raster/HDF work because of monkeypatches)

# Instantiate simulation in HECRAS-only mode (simulation will preload HECRAS KDTree)
sim = simulation(model_dir='.', model_name='test', crs='EPSG:32604', basin='Nushagak River',
                 water_temp=8.0, start_polygon=str(start_shp), env_files=env_files,
                 longitudinal_profile=str(long_shp), fish_length=500, num_timesteps=10, num_agents=50,
                 hecras_plan_path=str(plan), hecras_fields=['Cells Minimum Elevation', 'Water Surface',
                                                          'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y'],
                 hecras_k=8, use_hecras=True)

# Preload and time the mapping KDTree build
print('Preloading HECRAS map...')
start = time.time()
_m = load_hecras_plan_cached(sim, sim.hecras_plan_path, field_names=sim.hecras_fields)
print('KDTree built and fields loaded in %.3fs' % (time.time() - start))

# run environment for a few timesteps and time mapping
for t in range(5):
    t0 = time.time()
    sim.environment()
    dt = time.time() - t0
    print(f'Timestep {t}: environment() took {dt:.3f}s; sample depth[0]={sim.depth[0]:.3f}, x_vel[0]={sim.x_vel[0]:.3f}')

print('Done')
