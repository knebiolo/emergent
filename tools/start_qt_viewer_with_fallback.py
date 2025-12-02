"""Launcher for the RL Qt viewer with an import-time fallback.

If `sockeye_SoA_OpenGL_RL` fails to import, this script creates a lightweight
`DummySimulation` and `DummyTrainer` that implement the minimal API the viewer
needs so the UI can be inspected.
"""
import sys
import os
from PyQt5 import QtWidgets

try:
    from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights, PID_controller
    module_ok = True
except Exception as e:
    print('Could not import RL module:', e)
    module_ok = False

# Minimal dummy classes
class DummySimulation:
    def __init__(self, **kwargs):
        self.num_agents = kwargs.get('num_agents', 100)
        self.X = np.random.rand(self.num_agents) * 100
        self.Y = np.random.rand(self.num_agents) * 100
        self.heading = np.random.rand(self.num_agents) * 2 * np.pi
        self.x_vel = np.zeros(self.num_agents)
        self.y_vel = np.zeros(self.num_agents)
        self.dead = np.zeros(self.num_agents, dtype=int)
        self.hdf5 = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    def apply_behavioral_weights(self, bw):
        pass
    def reset_spatial_state(self):
        pass

class DummyTrainer:
    def __init__(self, sim):
        self.behavioral_weights = BehavioralWeights() if module_ok else type('BW', (), {'use_sog': True, 'sog_weight': 0.5})()

# Fallback viewer invocation
from tools.train_behavioral_weights_qt import RLTrainingViewer

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    if module_ok:
        # construct real sim via simulation() factory
        sim = simulation(num_agents=500)
        trainer = RLTrainer(sim)
        pid = PID_controller(sim.num_agents)
        hecras_plan = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    else:
        import numpy as np
        sim = DummySimulation(num_agents=500)
        trainer = DummyTrainer(sim)
        pid = None
        hecras_plan = os.path.join(os.path.dirname(__file__), '..', 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')

    viewer = RLTrainingViewer(sim, trainer, timesteps=1000, pid=pid, hecras_plan=hecras_plan, use_gl=False, show_raster=False)
    viewer.show()
    sys.exit(app.exec_())
