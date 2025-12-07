import sys, os, time
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.train_behavioral_weights_visual import setup_training_simulation
from emergent.salmon_abm.salmon_viewer_v2 import SalmonViewer
from PyQt5 import QtWidgets
from types import SimpleNamespace

args = SimpleNamespace()
args.fish_length = 450
args.timesteps = 1800
args.agents = 100
args.dt = 0.1

sim, trainer, hecras_plan = setup_training_simulation(args)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
viewer = SalmonViewer(sim, dt=args.dt, T=args.timesteps*args.dt, rl_trainer=trainer)

start = time.time()
timeout = 40.0
while time.time() - start < timeout:
    app.processEvents()
    if getattr(viewer, 'last_mesh_payload', None) is not None:
        print('PAYLOAD_READY')
        payload = viewer.last_mesh_payload
        print('verts=', payload['verts'].shape[0], 'faces=', payload['faces'].shape[0])
        break
    time.sleep(0.1)
else:
    print('TIMEOUT_NO_PAYLOAD')

# cleanup
try:
    viewer.close()
except Exception:
    pass

