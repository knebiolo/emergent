"""Non-blocking diagnostic: build SalmonViewer, call setup_background(), wait for mesh payload, print statuses."""
import time
import sys
import os
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from emergent.salmon_abm.salmon_viewer_v2 import SalmonViewer
from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation

# Minimal sim config: reuse training launcher discovery
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
print('Viewer instantiated')
viewer.setup_background()
print('setup_background called; waiting for mesh payload...')
# wait up to 10s for payload
for i in range(20):
    time.sleep(0.5)
    payload = getattr(viewer, 'last_mesh_payload', None)
    print(f'wait {i}: last_mesh_payload present={payload is not None}')
    if payload is not None:
        print('Payload keys:', list(payload.keys()))
        break
print('Diagnostic complete')
