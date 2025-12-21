"""Simple boids-like demo that updates agent positions and pushes them to the viewer.

This demo uses the `SalmonViewer` shim which now prefers the pyqtgraph adapter.
"""
import sys
import numpy as np
from PyQt5.QtCore import QTimer

from emergent.salmon_abm.viewer_v3.viewer_shim import SalmonViewer
from PyQt5 import QtWidgets


class SimpleSim:
    def __init__(self, n=50, bounds=1.0):
        self.n = n
        self.bounds = bounds
        # positions in NDC (-1..1)
        self.pos = (np.random.rand(n, 2) * 2.0 - 1.0) * 0.6
        angles = np.random.rand(n) * 2 * np.pi
        self.vel = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * 0.01

    def step(self):
        # simple wraparound / bounce
        self.pos += self.vel
        mask = self.pos > 1.0
        self.pos[mask] = -1.0
        mask2 = self.pos < -1.0
        self.pos[mask2] = 1.0


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim = SimpleSim(n=80)
    viewer = SalmonViewer(simulation=sim)
    # create a small background mesh (optional)
    verts = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype='f4')
    faces = np.array([[0,1,2],[2,3,0]], dtype='i4')
    colors = np.array([[0.2,0.4,0.6],[0.2,0.4,0.6],[0.2,0.4,0.6],[0.2,0.4,0.6]], dtype='f4')
    try:
        viewer.load_tin_payload({'verts': verts, 'faces': faces, 'colors': colors})
    except Exception:
        pass

    # start viewer (this will create UI and start the Qt event loop)
    def tick():
        sim.step()
        # pass positions as NDC XY (shape N,2)
        try:
            viewer.gl_widget.set_agents(sim.pos, size=3.0)
        except Exception:
            # older widgets may be nested
            try:
                viewer.gl_widget._cpu_widget.set_agents(sim.pos, size=3.0)
            except Exception:
                pass
    # attach timer to viewer to keep it alive with the window
    timer = QTimer(viewer)
    timer.timeout.connect(tick)
    timer.start(30)

    # run through the viewer shim which returns the QApplication exec result
    try:
        viewer.run()
    except Exception:
        # fallback: run QApplication directly
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
