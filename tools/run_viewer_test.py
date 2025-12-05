import sys
import numpy as np
from PyQt5.QtWidgets import QApplication

from emergent.salmon_abm.salmon_viewer import SalmonViewer

class DummySim:
    def __init__(self):
        self.num_agents = 10
        self.X = np.linspace(0, 50, self.num_agents)
        self.Y = np.zeros(self.num_agents)
        self.heading = np.zeros(self.num_agents)
        self.dead = np.zeros(self.num_agents, dtype=int)
        self.use_hecras = False
        self.vert_exag = 1.0
    def timestep(self, t, dt, g, pid):
        # simple forward motion for visual
        self.X += 0.1 * np.cos(self.heading)
        self.Y += 0.1 * np.sin(self.heading)
    def reset_spatial_state(self):
        self.X = np.linspace(0, 50, self.num_agents)
        self.Y = np.zeros(self.num_agents)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    sim = DummySim()
    viewer = SalmonViewer(sim, dt=0.1, T=10.0)
    viewer.show()
    print('Viewer created, entering Qt exec loop')
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print('Exception during app.exec_():', e)
        raise
