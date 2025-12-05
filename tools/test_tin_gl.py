"""Quick test for GL TIN rendering pipeline using synthetic points.

This script creates a minimal SalmonViewer-like object and calls the background
background setup that produces a TIN and launches the GL mesh builder. It does
not start the full event loop; it tests only that the GL view is created and
that the builder thread emits mesh_ready without raising immediate errors.
"""
import time
import numpy as np
from emergent.salmon_abm.salmon_viewer import SalmonViewer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

class DummySim:
    def __init__(self):
        # synthetic wetted nodes in a grid
        xs = np.linspace(0, 100, 40)
        ys = np.linspace(0, 60, 24)
        xv, yv = np.meshgrid(xs, ys)
        coords = np.column_stack([xv.ravel(), yv.ravel()])
        depths = np.sin(xv.ravel() / 10.0) + np.cos(yv.ravel() / 12.0) + 5.0
        self._hecras_geometry_info = {'perimeter_points': coords[:100].tolist()}
        self.num_agents = 0
        # fields used by viewer
        self.tin_thin_resolution = 5.0
        self.tin_max_nodes = 2000
        self.tin_alpha = None

        # hack: viewer expects 'hdf' for fallback; provide minimal mapping
        self.hdf = {}

dummy = DummySim()
app = QApplication([])
viewer = SalmonViewer(simulation=dummy, dt=0.1, T=10)
print('Viewer created. If GL is available, a GLViewWidget should be present or builder running.')

# Directly render a TIN from our synthetic arrays (calls background mesh builder)
# create a full 2D grid (not a diagonal line) so Delaunay has a full-dimensional input
xs = np.linspace(0, 100, 80)
ys = np.linspace(0, 60, 48)
xx, yy = np.meshgrid(xs, ys)
coords = np.column_stack([xx.ravel(), yy.ravel()])
vals = np.sin(xx.ravel() / 10.0) + np.cos(yy.ravel() / 12.0) + 5.0
viewer.render_tin_from_arrays(coords, vals, cap_nodes=4000)

# Run the Qt event loop briefly so the background thread can emit and the GL view can be created
QTimer.singleShot(2000, app.quit)  # quit after 2 seconds
app.exec_()

print('Event loop exited')
print('Has gl_view:', hasattr(viewer, 'gl_view'))
print('Has tin_mesh:', hasattr(viewer, 'tin_mesh'))
print('Done test — if a GL window appeared, close it manually.')
