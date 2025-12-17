"""Compatibility shim preserving `SalmonViewer` public API while using viewer_v3 internals.

This module provides a minimal `SalmonViewer` class and `launch_viewer` that
match the original signatures but delegate mesh building to `viewer_v3.mesh_builder`.
UI and GL rendering are intentionally unchanged at this scaffold stage; this shim
exists to make transitions safer and allow incremental migration.
"""
from typing import Any
import numpy as np
from PyQt5 import QtWidgets

from emergent.salmon_abm.viewer_v3 import mesh_builder


class SalmonViewer(QtWidgets.QWidget):
    def __init__(self, simulation: Any, dt: float = 0.1, T: int = 600, rl_trainer=None, **kwargs):
        super().__init__()
        self.sim = simulation
        self.dt = dt
        self.T = T
        self.rl_trainer = rl_trainer
        # Minimal UI placeholder to preserve API; full UI will be migrated later.
        self.setWindowTitle('SalmonViewer (v3 shim)')

    def setup_background(self):
        """Example usage of mesh_builder to create a mesh for preview/testing.

        This keeps the public behavior of computing a TIN but doesn't touch GL here.
        """
        perim_pts = getattr(self.sim, 'perimeter_points', None)
        if perim_pts is None:
            return None
        coords = np.asarray(perim_pts, dtype=float)
        vals = np.zeros(coords.shape[0], dtype=float)
        return mesh_builder.build_mesh(coords, vals, vert_exag=getattr(self.sim, 'vert_exag', 1.0))

    def run(self):
        self.show()
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
