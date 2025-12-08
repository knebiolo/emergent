"""Launch the original `salmon_viewer.py` for visual testing.
"""
import sys
import os
import time
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from emergent.salmon_abm.salmon_viewer import SalmonViewer
from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation

hecras_folder = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506')
hecras_plan = None
for f in os.listdir(hecras_folder):
    if f.endswith('.p05.hdf'):
        hecras_plan = os.path.join(hecras_folder, f)
        break
if hecras_plan is None:
    print('No HECRAS plan found; aborting')
    sys.exit(1)

config = {
    'model_dir': os.path.join(REPO_ROOT, 'outputs', 'diag'),
    'model_name': 'diag',
    'crs': 'EPSG:26904',
    'basin': 'Nushagak',
    'water_temp': 10.0,
    'start_polygon': os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'starting_location', 'start_loc_river_right.shp'),
    'centerline': os.path.join(REPO_ROOT, 'data', 'salmon_abm', 'Longitudinal', 'longitudinal.shp'),
    'fish_length': 450,
    'num_timesteps': 10,
    'num_agents': 1,
    'use_gpu': False,
    'defer_hdf': True,
    'hecras_plan_path': hecras_plan,
    'use_hecras': True,
    'hecras_k': 1,
}
os.makedirs(config['model_dir'], exist_ok=True)
sim = simulation(**config)
print('Sim created; pid_controller=', hasattr(sim, 'pid_controller'))

from PyQt5 import QtWidgets
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

viewer = SalmonViewer(sim, dt=0.1, T=10, rl_trainer=None)
print('Original viewer instantiated')
viewer.setup_background()
print('setup_background called; viewer should appear')
viewer.show()
app.exec_()
