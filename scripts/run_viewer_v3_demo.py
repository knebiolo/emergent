"""Demo runner for viewer_v3 moderngl renderer.

Run with: python scripts/run_viewer_v3_demo.py
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore

try:
    from emergent.salmon_abm.viewer_v3.renderer_cpu_fast import FastCPUViewerWidget as ModernglViewerWidget
except Exception:
    from emergent.salmon_abm.viewer_v3.renderer_cpu import CPUViewerWidget as ModernglViewerWidget
from emergent.salmon_abm.viewer_v3.mesh_builder import build_mesh


class DemoApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = ModernglViewerWidget(self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.viewer)
        self.setLayout(layout)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(1000 // 30)  # target ~30 FPS
        # create synthetic points in a grid
        gx, gy = np.mgrid[0:10:25j, 0:6:15j]
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        vals = np.sin(pts[:, 0] * 0.3) * np.cos(pts[:, 1] * 0.25)
        verts, faces, colors = build_mesh(pts, vals, vert_exag=1.0)
        self.viewer.set_mesh(verts, faces, colors, vert_exag=1.0)

    def step(self):
        # simple animation by modifying Z in CPU and re-uploading color/z
        if self.viewer.verts is None:
            return
        t = QtCore.QTime.currentTime().msec() / 1000.0
        verts = self.viewer.verts.copy()
        verts[:, 2] = verts[:, 2] + 0.2 * np.sin(t + verts[:, 0] * 0.1)
        # keep same colors
        colors = self.viewer.colors if getattr(self.viewer, 'colors', None) is not None else np.ones((verts.shape[0], 4), dtype='f4')
        faces = self.viewer.faces if getattr(self.viewer, 'faces', None) is not None else np.zeros((0,3), dtype='i4')
        self.viewer.set_mesh(verts, faces, colors, vert_exag=1.0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = DemoApp()
    w.show()
    sys.exit(app.exec_())
