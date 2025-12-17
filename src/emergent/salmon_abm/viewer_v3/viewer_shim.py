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
from emergent.salmon_abm.viewer_v3.renderer_moderngl import ModernglViewerWidget
from emergent.salmon_abm.viewer_v3.realtime import RealTimeSolver
from PyQt5.QtWidgets import QPushButton, QLabel, QSlider, QGroupBox, QCheckBox, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt


class SalmonViewer(QtWidgets.QWidget):
    def __init__(self, simulation: Any, dt: float = 0.1, T: int = 600, rl_trainer=None, **kwargs):
        super().__init__()
        self.sim = simulation
        self.dt = dt
        self.T = T
        self.rl_trainer = rl_trainer
        # Minimal UI placeholder to preserve API; full UI will be migrated later.
        self.setWindowTitle('SalmonViewer (v3 shim)')
        # create moderngl widget
        try:
            self.gl_widget = ModernglViewerWidget(self)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.gl_widget)
            self.setLayout(layout)
        except Exception:
            self.gl_widget = None
        # runtime solver (not started by default)
        self._rt_solver = None
        # create control widgets (right panel)
        try:
            self.play_btn = QPushButton('Play')
            self.play_btn.clicked.connect(self._on_play)
            self.pause_btn = QPushButton('Pause')
            self.pause_btn.clicked.connect(self._on_pause)
            self.reset_btn = QPushButton('Reset')
            self.reset_btn.clicked.connect(self._on_reset)

            self.rebuild_btn = QPushButton('Rebuild Background')
            self.rebuild_btn.clicked.connect(self.setup_background)

            self.ve_label = QLabel('Z Exag: 1.00x')
            self.ve_slider = QSlider(Qt.Horizontal)
            self.ve_slider.setMinimum(1)
            self.ve_slider.setMaximum(500)
            self.ve_slider.setValue(100)
            self.ve_slider.valueChanged.connect(self._on_ve_changed)

            self.show_dead_cb = QCheckBox('Show Dead')
            self.show_dead_cb.setChecked(False)
            self.show_direction_cb = QCheckBox('Show Direction')
            self.show_direction_cb.setChecked(False)
        except Exception:
            pass

    def setup_background(self):
        """Example usage of mesh_builder to create a mesh for preview/testing.

        This keeps the public behavior of computing a TIN but doesn't touch GL here.
        """
        # Prefer extracting a raster from simulation.hdf5 if available
        try:
            if hasattr(self.sim, 'hdf5') and self.sim.hdf5 is not None and 'environment' in self.sim.hdf5:
                env = self.sim.hdf5['environment']
                if 'depth' in env:
                    depth = np.asarray(env['depth'])
                    bbox = getattr(self.sim, 'depth_rast_bbox', None)
                    if self.gl_widget is not None:
                        self.gl_widget.set_heightmap(depth, bbox=bbox, vert_exag=getattr(self.sim, 'vert_exag', 1.0))
                        return True
        except Exception:
            pass
        # fallback to perim points -> mesh builder
        perim_pts = getattr(self.sim, 'perimeter_points', None)
        if perim_pts is None:
            return None
        coords = np.asarray(perim_pts, dtype=float)
        vals = np.zeros(coords.shape[0], dtype=float)
        verts, faces, colors = mesh_builder.build_mesh(coords, vals, vert_exag=getattr(self.sim, 'vert_exag', 1.0))
        if self.gl_widget is not None:
            self.gl_widget.set_mesh(verts, faces, colors)
        return True

    def start_realtime(self, target_fps: int = 30):
        if self._rt_solver is not None:
            return
        try:
            self._rt_solver = RealTimeSolver(self.sim, dt=self.dt, target_fps=target_fps)
            self._rt_solver.frame_ready.connect(self._on_frame_ready)
            self._rt_solver.start()
        except Exception:
            self._rt_solver = None

    def stop_realtime(self):
        try:
            if self._rt_solver is not None:
                self._rt_solver.stop()
                self._rt_solver = None
        except Exception:
            pass

    def _on_frame_ready(self, payload: dict):
        try:
            pos = payload.get('positions', None)
            if pos is not None and self.gl_widget is not None:
                # use default color and upload
                self.gl_widget.set_agents(pos)
        except Exception:
            pass

    def _on_play(self):
        # start realtime solver
        self.start_realtime(target_fps=30)

    def _on_pause(self):
        if self._rt_solver is not None:
            self._rt_solver.pause()

    def _on_reset(self):
        try:
            if hasattr(self.sim, 'reset_spatial_state'):
                self.sim.reset_spatial_state(reset_positions=True)
        except Exception:
            pass

    def _on_ve_changed(self, v: int):
        try:
            ex = v / 100.0
            self.ve_label.setText(f'Z Exag: {ex:.2f}x')
            if self.gl_widget is not None:
                # rebuild background with new exaggeration
                self.setup_background()
        except Exception:
            pass

    def run(self):
        # arrange a three-column layout: left (metrics) | center (gl) | right (controls)
        main = QHBoxLayout()
        # left metrics placeholder
        left = QGroupBox('Metrics')
        left_layout = QVBoxLayout()
        self.mean_speed_label = QLabel('Mean Speed: --')
        left_layout.addWidget(self.mean_speed_label)
        left.setLayout(left_layout)

        center_layout = QVBoxLayout()
        if self.gl_widget is not None:
            center_layout.addWidget(self.gl_widget)

        right = QGroupBox('Controls')
        right_layout = QVBoxLayout()
        try:
            right_layout.addWidget(self.play_btn)
            right_layout.addWidget(self.pause_btn)
            right_layout.addWidget(self.reset_btn)
            right_layout.addWidget(self.rebuild_btn)
            right_layout.addWidget(self.ve_label)
            right_layout.addWidget(self.ve_slider)
            right_layout.addWidget(self.show_dead_cb)
            right_layout.addWidget(self.show_direction_cb)
        except Exception:
            pass
        right.setLayout(right_layout)

        main.addWidget(left, 1)
        main.addLayout(center_layout, 4)
        main.addWidget(right, 1)
        self.setLayout(main)

        self.show()
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
