"""
salmon_viewer_v2.py

A clean, minimal replacement viewer module to be used while we repair the
original `salmon_viewer.py`. This file contains a single-threaded GL mesh
builder (QThread) and a compact `SalmonViewer` QWidget with `launch_viewer`.

Purpose: provide a working baseline so you can run the RL visual training GUI
and verify GL TIN rendering without the mangled original file.
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QSlider, QGroupBox

import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


class _GLMeshBuilder(QtCore.QThread):
    mesh_ready = QtCore.pyqtSignal(object)

    def __init__(self, pts, vals, vert_exag=1.0, parent=None):
        super().__init__(parent=parent)
        self.pts = np.asarray(pts, dtype=float)
        self.vals = np.asarray(vals, dtype=float)
        self.vert_exag = float(vert_exag)

    def run(self):
        try:
            # compute triangulation in the worker thread to avoid blocking the UI
            try:
                from scipy.spatial import Delaunay
                tri = Delaunay(self.pts)
                tris = tri.simplices
            except Exception as e:
                self.mesh_ready.emit({"error": e})
                return

            if self.pts.shape[0] == self.vals.shape[0]:
                z = self.vals * self.vert_exag
                verts = np.column_stack([self.pts[:, 0], self.pts[:, 1], z])
            else:
                verts = np.column_stack([self.pts[:, 0], self.pts[:, 1], np.zeros(len(self.pts))])

            # simple color mapping
            try:
                zvals = verts[:, 2]
                vmin, vmax = np.nanmin(zvals), np.nanmax(zvals)
                span = vmax - vmin if vmax > vmin else 1.0
                norm = (zvals - vmin) / span
                colors = np.zeros((len(verts), 4), dtype=float)
                colors[:, 0] = 0.2 + 0.8 * norm
                colors[:, 1] = 0.4 * (1.0 - norm)
                colors[:, 2] = 0.6 * (1.0 - norm)
                colors[:, 3] = 1.0
            except Exception:
                colors = np.tile([0.6, 0.6, 0.6, 1.0], (len(verts), 1))

            self.mesh_ready.emit({"verts": verts.astype(float), "faces": tris.astype(int), "colors": colors.astype(float)})
        except Exception as e:
            self.mesh_ready.emit({"error": e})


class SalmonViewer(QtWidgets.QWidget):
    def __init__(self, sim, dt=0.1, T=600, rl_trainer=None, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.sim = sim
        self.dt = dt
        self.T = T
        self.rl_trainer = rl_trainer

        self.current_timestep = 0
        self.paused = True

        self.tin_mesh = None
        self.gl_view = None

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(self.dt * 1000))

        QTimer.singleShot(10, self.setup_background)

    def _build_ui(self):
        self.setWindowTitle('SalmonViewer v2')
        main = QHBoxLayout(self)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(600, 600)
        main.addWidget(self.plot_widget, 3)

        right = QWidget()
        rlay = QVBoxLayout(right)
        btn_play = QPushButton('Play')
        btn_play.clicked.connect(self._on_play)
        btn_pause = QPushButton('Pause')
        btn_pause.clicked.connect(self.toggle_pause)
        btn_reset = QPushButton('Reset')
        btn_reset.clicked.connect(self.reset_simulation)
        rlay.addWidget(btn_play)
        rlay.addWidget(btn_pause)
        rlay.addWidget(btn_reset)

        ve_group = QGroupBox('Vertical Exaggeration')
        ve_layout = QVBoxLayout()
        self.ve_label = QLabel('Z Exag: 1.00x')
        ve_layout.addWidget(self.ve_label)
        self.ve_slider = QSlider(Qt.Horizontal)
        self.ve_slider.setMinimum(1)
        self.ve_slider.setMaximum(500)
        self.ve_slider.setValue(100)
        self.ve_slider.valueChanged.connect(lambda v: self.ve_label.setText(f'Z Exag: {v/100:.2f}x'))
        ve_layout.addWidget(self.ve_slider)
        ve_group.setLayout(ve_layout)
        rlay.addWidget(ve_group)

        rlay.addStretch()
        main.addWidget(right, 1)

    def _on_play(self):
        self.paused = False

    def setup_background(self):
        try:
            if not (hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path')):
                print('[RENDER] No HECRAS configured')
                return
            import h5py
            plan = self.sim.hecras_plan_path
            with h5py.File(plan, 'r') as hdf:
                coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
            mask = depth > 0.05
            pts = coords[mask]
            vals = depth[mask]
            max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
            if len(pts) > max_nodes:
                idx = np.random.default_rng(0).choice(len(pts), size=max_nodes, replace=False)
                pts = pts[idx]; vals = vals[idx]
            if len(pts) < 3:
                print('[RENDER] Not enough points for TIN')
                return
            # Triangulation will be computed in the background builder thread

            if gl is None:
                print('[RENDER] GL not available')
                return

            if self.gl_view is None:
                self.gl_view = gl.GLViewWidget()
                parent = self.plot_widget.parent()
                layout = parent.layout()
                layout.removeWidget(self.plot_widget)
                self.plot_widget.hide()
                layout.addWidget(self.gl_view)

            builder = _GLMeshBuilder(pts, vals, vert_exag=getattr(self.sim, 'vert_exag', 1.0), parent=self)

            def _on_mesh(payload):
                if 'error' in payload:
                    print('[TIN] builder error', payload['error'])
                    return
                # expose latest payload for external tests/inspection
                try:
                    self.last_mesh_payload = payload
                except Exception:
                    pass
                # payload received and stored at `self.last_mesh_payload`
                verts = payload['verts']; faces = payload['faces']; colors = payload['colors']
                meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded')
                if self.tin_mesh is not None:
                    try:
                        self.gl_view.removeItem(self.tin_mesh)
                    except Exception:
                        pass
                self.tin_mesh = mesh
                self.gl_view.addItem(mesh)

            builder.mesh_ready.connect(_on_mesh)
            try:
                builder.start()
            except Exception:
                builder.run()
        except Exception as e:
            print('[RENDER] setup_background failed:', e)

    def update_simulation(self):
        if self.paused:
            return
        try:
            # Require the simulation to provide a PID controller. No fallback.
            pid = getattr(self.sim, 'pid_controller', None)
            if pid is None:
                raise RuntimeError('Simulation has no pid_controller; ensure the sim creates it before visualization (no fallback allowed)')

            self.sim.timestep(self.current_timestep, self.dt, 9.81, pid)
            self.current_timestep += 1
            self.update_displays()
        except Exception as e:
            print('[ERROR] simulation step failed:', e)
            self.paused = True

    def update_displays(self):
        if gl is None or self.gl_view is None:
            return
        if not (hasattr(self.sim, 'X') and hasattr(self.sim, 'Y')):
            return
        alive = (getattr(self.sim, 'dead', np.zeros(getattr(self.sim, 'num_agents', 0))) == 0)
        if getattr(self.sim, 'num_agents', 0) == 0:
            return
        x = self.sim.X[alive]; y = self.sim.Y[alive]
        pts = np.column_stack([x, y, np.zeros_like(x)])
        scatter = gl.GLScatterPlotItem(pos=pts, color=(1.0, 0.4, 0.4, 0.9), size=6)
        try:
            if hasattr(self, 'gl_agent_scatter') and self.gl_agent_scatter is not None:
                self.gl_view.removeItem(self.gl_agent_scatter)
        except Exception:
            pass
        self.gl_agent_scatter = scatter
        self.gl_view.addItem(self.gl_agent_scatter)

    def toggle_pause(self):
        self.paused = not self.paused

    def reset_simulation(self):
        try:
            if hasattr(self.sim, 'reset_spatial_state'):
                try:
                    self.sim.reset_spatial_state(reset_positions=True)
                except TypeError:
                    self.sim.reset_spatial_state()
        except Exception as e:
            print('[RESET] Exception:', e)
        self.current_timestep = 0
        self.paused = True

    def run(self):
        self.show()
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
