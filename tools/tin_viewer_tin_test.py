import argparse
import time
from types import SimpleNamespace

# Ensure we import from repo root

# Build a minimal args object similar to CLI
args = SimpleNamespace()
args.fish_length = 450
args.timesteps = 10
args.agents = 10
args.dt = 0.1

# Import setup from the training helper to create a simulation instance
import sys, os
# ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from tools.train_behavioral_weights_visual import setup_training_simulation
sim, trainer, hecras_plan = setup_training_simulation(args)

# Import Qt and the viewer
from PyQt5 import QtWidgets
from emergent.salmon_abm.salmon_viewer import SalmonViewer

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

viewer = SalmonViewer(sim, dt=args.dt, T=args.timesteps * args.dt, rl_trainer=trainer)

# Call setup_background directly and allow some event processing so the builder thread runs
viewer.setup_background()

start = time.time()
# process events for up to 30 seconds to let the TIN builder finish, print progress
wait = 30.0
next_print = 1.0
while time.time() - start < wait:
    app.processEvents()
    time.sleep(0.05)
    elapsed = time.time() - start
    if elapsed >= next_print:
        print(f'TIN wait: {int(elapsed)}s (wall {time.strftime("%H:%M:%S")})')
        next_print += 1.0

print('TIN test complete')

# Clean up
try:
    viewer.close()
except Exception:
    pass

# Exit

if getattr(viewer, 'last_mesh_payload', None) is not None:
    print('TIN mesh created (payload)')
else:
    print('TIN mesh NOT created')

