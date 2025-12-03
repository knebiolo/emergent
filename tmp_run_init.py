import os
import sys
repo_root = os.path.dirname(__file__)
sys.path.insert(0, repo_root)
from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import SockeyeOpenGLSimulation

# minimal kwargs — avoid HECRAS unless file exists
kwargs = {
    'num_agents': 100,
    'num_timesteps': 10,
    'hecras_plan_path': None,
    'use_hecras': False,
    'longitudinal_profile': None,
}

print('Instantiating simulation...')
try:
    sim = SockeyeOpenGLSimulation(**kwargs)
    print('Sim instantiated')
    cl = getattr(sim, 'longitude', None)
    print('longitude:', type(cl), cl)
    cl2 = getattr(sim, 'centerline', None)
    print('centerline:', type(cl2), cl2)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
