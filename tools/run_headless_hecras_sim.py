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

# Monkeypatch the class to skip raster imports during __init__ (we'll rely on HECRAS mapping)
sock_mod.simulation.enviro_import = lambda self, a, b: None
# also bypass boundary_surface and batch_sample_environment during init to avoid HDF/raster dependencies
sock_mod.simulation.boundary_surface = lambda self: None
sock_mod.simulation.batch_sample_environment = lambda self, transforms, names: {n: np.zeros(self.num_agents) for n in names}
# bypass mental/refugia initialization that depends on raster size during init
sock_mod.simulation.initialize_mental_map = lambda self: None
sock_mod.simulation.initialize_refugia_map = lambda self: None
    
# monkeypatch initial_heading to avoid reading vel_dir during init
def _init_heading_stub(self):
    self.heading = np.zeros(self.num_agents)
    self.max_practical_sog = np.stack((self.sog * np.cos(self.heading), self.sog * np.sin(self.heading)))

sock_mod.simulation.initial_heading = _init_heading_stub

# monkeypatch initial_swim_speed to avoid sampling rasters during init
def _init_swim_speed_stub(self):
    # set default x_vel,y_vel to zero arrays if missing
    if not hasattr(self, 'x_vel'):
        self.x_vel = np.zeros(self.num_agents)
    if not hasattr(self, 'y_vel'):
        self.y_vel = np.zeros(self.num_agents)
    self.swim_speed = np.zeros(self.num_agents)

sock_mod.simulation.initial_swim_speed = _init_swim_speed_stub

# Instantiate simulation with small agent count
# Instantiate simulation (init will skip heavy raster/HDF work because of monkeypatches)
sim = simulation(model_dir='.', model_name='test', crs='EPSG:32604', basin='Nushagak River',
                 water_temp=8.0, start_polygon=str(start_shp), env_files=env_files,
                 longitudinal_profile=str(long_shp), fish_length=500, num_timesteps=10, num_agents=50)

# provide minimal raster dimensions expected by some methods
sim.width = 1024
sim.height = 1024

# Monkeypatch heavy raster import to no-op because we'll use HECRAS mapping
sim.enviro_import = types.MethodType(lambda self, a, b: None, sim)

# Override initial env attributes that enviro_import would normally set
sim.depth_rast_transform = None
sim.vel_x_rast_transform = None
sim.vel_y_rast_transform = None
sim.vel_mag_rast_transform = None
sim.wetted_transform = None

# Set HECRAS mapping fields and plan
sim.hecras_plan_path = str(plan)
sim.hecras_fields = ['Cells Minimum Elevation', 'Water Surface', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y']
sim.hecras_k = 8

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
