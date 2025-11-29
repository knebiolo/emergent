"""Create a minimal simulation object and call its environment() to test HECRAS mapping integration."""
import types
import numpy as np
from pathlib import Path
from emergent.salmon_abm.sockeye_SoA import HECRASMap

# We'll create a minimal object with the attributes used by environment()
SimClass = type('SimClass', (), {})
s = SimClass()
# minimal agent count
s.num_agents = 10
# random agent positions (near valid coords)
plan = Path(r"c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\emergent\data\salmon_abm\20240506\Nuyakuk_Production_.p05.hdf")
print('Plan:', plan)
# build a HECRASMap and sample to get coords bounds
m = HECRASMap(str(plan), field_names=['Cells Minimum Elevation'])
# pick some coords
rng = np.random.default_rng(0)
idx = rng.choice(m.coords.shape[0], size=s.num_agents, replace=False)
pts = m.coords[idx]
s.X = pts[:,0].copy()
s.Y = pts[:,1].copy()
# simple raster transforms placeholders used by batch_sample_environment — we won't call that here
s.depth_rast_transform = None
s.vel_x_rast_transform = None
s.vel_y_rast_transform = None
s.vel_mag_rast_transform = None
s.wetted_transform = None
s.longitudinal = None
# set hecras config
s.hecras_plan_path = str(plan)
s.hecras_fields = ['Cells Minimum Elevation', 'Water Surface', 'Cell Velocity - Velocity X', 'Cell Velocity - Velocity Y']
s.hecras_k = 8

# add dummy batch_sample_environment that returns zeros (environment will then be overwritten by HECRAS mapping)
def dummy_batch_sample(transforms, names):
    N = s.num_agents
    out = {n: np.zeros(N, dtype=float) for n in names}
    return out

s.batch_sample_environment = dummy_batch_sample
# add minimal compute_linear_positions
s.compute_linear_positions = lambda x: np.zeros(s.num_agents,)

# bind environment method from module
import importlib
mod = importlib.import_module('emergent.salmon_abm.sockeye_SoA')
env_fn = getattr(mod, 'simulation').environment
s.environment = types.MethodType(env_fn, s)

# run
s.environment()
print('depth sample:', s.depth[:5])
print('wsel sample:', getattr(s,'wsel',None)[:5])
print('x_vel sample:', s.x_vel[:5])
print('y_vel sample:', s.y_vel[:5])
print('Done')
