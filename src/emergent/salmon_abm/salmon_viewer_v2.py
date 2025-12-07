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
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QSlider, QGroupBox, QCheckBox
import os

import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None

try:
    from .tin_helpers import sample_evenly
except Exception:
    try:
        from emergent.salmon_abm.tin_helpers import sample_evenly
    except Exception:
        sample_evenly = None


class _GLMeshBuilder(QtCore.QThread):
    mesh_ready = QtCore.pyqtSignal(object)

    def __init__(self, pts, vals, vert_exag=1.0, poly=None, parent=None):
        super().__init__(parent=parent)
        self.pts = np.asarray(pts, dtype=float)
        self.vals = np.asarray(vals, dtype=float)
        self.vert_exag = float(vert_exag)
        self.poly = poly

    def run(self):
        try:
            # compute triangulation in the worker thread to avoid blocking the UI
            try:
                # Prefer the helper which triangulates and clips using the sim polygon
                try:
                    from emergent.salmon_abm.tin_helpers import triangulate_and_clip
                except Exception:
                    from .tin_helpers import triangulate_and_clip

                verts, tris = triangulate_and_clip(self.pts, self.vals * self.vert_exag if self.pts.shape[0] == self.vals.shape[0] else np.zeros(len(self.pts)), poly=self.poly)
            except Exception as e:
                self.mesh_ready.emit({"error": e})
                return

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
        # Start running immediately so agents swim over the TIN
        self.paused = False

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
        btn_perim = QPushButton('Toggle Perimeter')
        btn_perim.clicked.connect(self.toggle_perimeter)
        rlay.addWidget(btn_perim)
        btn_rebuild = QPushButton('Rebuild TIN')
        btn_rebuild.clicked.connect(self.rebuild_tin_action)
        rlay.addWidget(btn_rebuild)
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

        # Metrics placeholders (ported from original viewer)
        self.mean_speed_label = QLabel('Mean Speed: --')
        self.max_speed_label = QLabel('Max Speed: --')
        self.mean_energy_label = QLabel('Mean Energy: --')
        self.min_energy_label = QLabel('Min Energy: --')
        self.upstream_progress_label = QLabel('Upstream Progress: --')
        self.mean_centerline_label = QLabel('Mean Centerline: --')
        self.mean_passage_delay_label = QLabel('Mean Passage Delay: --')
        self.passage_success_rate_label = QLabel('Passage Success: --')

        # compact metrics layout at the bottom of the right pane
        try:
            metrics_box = QGroupBox('Metrics')
            metrics_layout = QVBoxLayout()
            metrics_layout.addWidget(self.mean_speed_label)
            metrics_layout.addWidget(self.max_speed_label)
            metrics_layout.addWidget(self.mean_energy_label)
            metrics_layout.addWidget(self.min_energy_label)
            metrics_layout.addWidget(self.upstream_progress_label)
            metrics_layout.addWidget(self.mean_centerline_label)
            metrics_layout.addWidget(self.mean_passage_delay_label)
            metrics_layout.addWidget(self.passage_success_rate_label)
            # small extras
            self.mean_nn_dist_label = QLabel('Mean NN Dist: --')
            self.polarization_label = QLabel('Polarization: --')
            metrics_layout.addWidget(self.mean_nn_dist_label)
            metrics_layout.addWidget(self.polarization_label)

            # per-episode metrics tracking helper (checkboxes)
            self._available_episode_metrics = [
                'collision_count', 'mean_upstream_progress', 'mean_upstream_velocity',
                'energy_efficiency', 'mean_passage_delay'
            ]
            self.track_metric_cbs = {}

            def add_label_with_cb(label_widget, metric_key, default_checked=False):
                h = QHBoxLayout()
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(6)
                h.addWidget(label_widget)
                cb = QCheckBox()
                cb.setChecked(default_checked)
                cb.setFixedWidth(22)
                h.addWidget(cb)
                metrics_layout.addLayout(h)
                self.track_metric_cbs[metric_key] = cb

            add_label_with_cb(self.mean_energy_label, 'energy_efficiency')
            add_label_with_cb(self.mean_passage_delay_label, 'mean_passage_delay')

            # per-episode plot
            self.per_episode_plot = pg.PlotWidget(title='Per-Episode Metrics')
            self.per_episode_plot.setLabel('bottom', 'Episode')
            self.per_episode_plot.setLabel('left', 'Metric Value')
            self.per_episode_plot.setMaximumHeight(220)
            metrics_layout.addWidget(self.per_episode_plot)

            metrics_box.setLayout(metrics_layout)
            rlay.addWidget(metrics_box)
        except Exception:
            pass

        # RL status labels
        try:
            self.episode_label = QLabel('Episode: 0 | Timestep: 0')
            self.reward_label = QLabel('Reward: 0.00')
            self.best_reward_label = QLabel('Best: 0.00')
            rlay.addWidget(self.episode_label)
            rlay.addWidget(self.reward_label)
            rlay.addWidget(self.best_reward_label)
        except Exception:
            pass

    def _on_play(self):
        self.paused = False

    def toggle_perimeter(self):
        try:
            if not hasattr(self, 'perim_visible'):
                self.perim_visible = True
            # toggle
            self.perim_visible = not self.perim_visible
            if getattr(self, 'perim_scatter', None) is None:
                return
            if self.perim_visible:
                try:
                    self.gl_view.addItem(self.perim_scatter)
                except Exception:
                    pass
            else:
                try:
                    self.gl_view.removeItem(self.perim_scatter)
                except Exception:
                    pass
        except Exception:
            pass

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
            # determine sampling parameters
            max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
            depth_thresh = getattr(self.sim, 'tin_depth_thresh', 0.05)

            mask = depth > depth_thresh
            pts = coords[mask]
            vals = depth[mask]

            # improved spatial sampling (if helper available)
            if sample_evenly is not None and len(pts) > max_nodes:
                pts, vals = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=120)
            elif len(pts) > max_nodes:
                idx = np.random.default_rng(0).choice(len(pts), size=max_nodes, replace=False)
                pts = pts[idx]; vals = vals[idx]

            if len(pts) < 3:
                print('[RENDER] Not enough points for TIN')
                return

            # Triangulation will be computed in the background builder thread
            # Use perimeter generated by the simulation (vector-first). Viewer no longer
            # performs its own dry-cell inference — sim is the single source of truth.
            try:
                sim_perim = getattr(self.sim, 'perimeter_points', None)
                if sim_perim is not None and getattr(sim_perim, 'shape', (0,))[0] > 0:
                    self.perimeter_pts = np.asarray(sim_perim)
                else:
                    # No perimeter available from sim; do not attempt local inference.
                    self.perimeter_pts = None
            except Exception:
                self.perimeter_pts = None

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

            # pass sim polygon so clipping happens in worker thread
            sim_poly = getattr(self.sim, 'perimeter_polygon', None)
            builder = _GLMeshBuilder(pts, vals, vert_exag=getattr(self.sim, 'vert_exag', 1.0), poly=sim_poly, parent=self)

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
                # clipping already handled in worker thread
                meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded')
                if self.tin_mesh is not None:
                    try:
                        self.gl_view.removeItem(self.tin_mesh)
                    except Exception:
                        pass
                self.tin_mesh = mesh
                self.gl_view.addItem(mesh)
                # fit camera to mesh extents
                try:
                    verts_min = np.min(verts[:, :2], axis=0)
                    verts_max = np.max(verts[:, :2], axis=0)
                    center = (verts_min + verts_max) / 2.0
                    size = np.max(verts_max - verts_min)
                    # place camera above center, distance based on size
                    self.gl_view.setCameraPosition(pos=QtCore.QVector3D(center[0], center[1], size*1.0), elevation=90, azimuth=0)
                except Exception:
                    pass
                # save a one-time screenshot for verification
                try:
                    out_dir = getattr(self.sim, 'model_dir', None) or '.'
                    out_path = os.path.join(out_dir, 'outputs', 'tin_screenshot.png')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    img = self.gl_view.readQImage()
                    img.save(out_path)
                    print(f'[TIN] screenshot saved: {out_path}')
                except Exception:
                    try:
                        # fallback: write a simple 2D matplotlib visualization of verts
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.triplot(verts[:, 0], verts[:, 1], faces, linewidth=0.2)
                        ax.set_aspect('equal')
                        ax.set_title('TIN preview (2D)')
                        fig.savefig(out_path, dpi=150)
                        plt.close(fig)
                        print(f'[TIN] fallback 2D screenshot saved: {out_path}')
                    except Exception:
                        pass

            builder.mesh_ready.connect(_on_mesh)
            try:
                builder.start()
            except Exception:
                builder.run()
        except Exception as e:
            print('[RENDER] setup_background failed:', e)

    def rebuild_tin_action(self):
        try:
            if hasattr(self, 've_slider'):
                self.sim.vert_exag = self.ve_slider.value() / 100.0
            QtCore.QTimer.singleShot(10, self.setup_background)
        except Exception as e:
            print('[REBUILD] Exception rebuilding TIN:', e)

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
        # if perimeter data exists and GL available, add perimeter overlay
        try:
            if getattr(self, 'perimeter_pts', None) is not None and gl is not None:
                perim = self.perimeter_pts
                perim_pts3 = np.column_stack([perim[:,0], perim[:,1], np.zeros(len(perim))])
                self.perim_scatter = gl.GLScatterPlotItem(pos=perim_pts3, color=(0.0, 0.8, 0.0, 1.0), size=2)
                self.gl_view.addItem(self.perim_scatter)
        except Exception:
            pass

    def toggle_pause(self):
        self.paused = not self.paused

    def refresh_rl_labels(self):
        try:
            self.episode_label.setText(f"Episode: {getattr(self, 'current_episode', 0)} | Timestep: {getattr(self, 'current_timestep', 0)}")
            self.reward_label.setText(f"Reward: {getattr(self, 'episode_reward', 0.0):.2f}")
            self.best_reward_label.setText(f"Best: {getattr(self, 'best_reward', 0.0):.2f}")
        except Exception:
            pass

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
        try:
            # attempt to raise and activate the window so it appears on top
            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
            print('[VIEWER] shown')
        except Exception:
            pass
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
