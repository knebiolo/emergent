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

    def load_tin_payload(self, payload_or_path):
        """Headless-friendly loader: accept a payload dict (as emitted by _GLMeshBuilder)
        or a `.npz`/`.npy` file path and return (verts, faces, colors).

        This function does not require OpenGL and can be used in headless tests.
        """
        # payload dict path
        if isinstance(payload_or_path, dict):
            payload = payload_or_path
            if 'error' in payload:
                raise RuntimeError(f"payload contains error: {payload['error']}")
            verts = payload.get('verts')
            faces = payload.get('faces')
            colors = payload.get('colors')
            return np.asarray(verts), np.asarray(faces), np.asarray(colors)

        # file path path
        path = str(payload_or_path)
        if os.path.exists(path):
            if path.endswith('.npz'):
                data = np.load(path)
                verts = data.get('verts')
                faces = data.get('faces')
                colors = data.get('colors')
                return np.asarray(verts), np.asarray(faces), np.asarray(colors)
            elif path.endswith('.npy'):
                verts = np.load(path)
                return np.asarray(verts), np.zeros((0, 3), dtype=int), np.zeros((len(verts), 4))
        raise FileNotFoundError(path)

    def _build_ui(self):
        self.setWindowTitle('SalmonViewer')
        main = QHBoxLayout(self)

        # Left panel: tools and metrics
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setMinimumWidth(260)

        # Center: main plot or GL view
        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(600, 600)
        center_layout.addWidget(self.plot_widget)

        # Right: RL controls and status
        right = QWidget()
        rlay = QVBoxLayout(right)
        # Playback controls (match original viewer ordering and widget names)
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self._on_play)
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_simulation)
        rlay.addWidget(self.play_btn)
        rlay.addWidget(self.pause_btn)
        rlay.addWidget(self.reset_btn)

        # Utility buttons (match original viewer ordering)
        btn_rebuild = QPushButton('Rebuild TIN')
        btn_rebuild.clicked.connect(self.rebuild_tin_action)
        rlay.addWidget(btn_rebuild)
        btn_perim = QPushButton('Toggle Perimeter')
        btn_perim.clicked.connect(self.toggle_perimeter)
        rlay.addWidget(btn_perim)

        # Small visual toggles to match original viewer
        try:
            self.show_dead_cb = QCheckBox('Show Dead')
            self.show_dead_cb.setChecked(False)
            rlay.addWidget(self.show_dead_cb)
            self.show_direction_cb = QCheckBox('Show Direction')
            self.show_direction_cb.setChecked(False)
            rlay.addWidget(self.show_direction_cb)
            self.show_tail_cb = QCheckBox('Show Tail')
            self.show_tail_cb.setChecked(False)
            rlay.addWidget(self.show_tail_cb)
        except Exception:
            pass

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

        # assemble main three-column layout: left | center | right
        main.addWidget(left, 1)
        main.addWidget(center, 3)
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

            # Map labels to metric keys (match original viewer)
            # collision_count label (new)
            self.collision_count_label = QLabel('Collision Count: --')
            add_label_with_cb(self.collision_count_label, 'collision_count')
            add_label_with_cb(self.upstream_progress_label, 'mean_upstream_progress')
            self.mean_upstream_velocity_label = QLabel('Mean Upstream Velocity: --')
            add_label_with_cb(self.mean_upstream_velocity_label, 'mean_upstream_velocity')
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
        try:
            self.play_btn.setText('Play')
            self.pause_btn.setText('Pause')
        except Exception:
            pass

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
        num_agents = getattr(self.sim, 'num_agents', 0)
        if num_agents == 0:
            return

        dead_mask = (getattr(self.sim, 'dead', np.zeros(num_agents)) != 0)
        alive_mask = ~dead_mask

        # Respect show-dead checkbox if present
        try:
            if getattr(self, 'show_dead_cb', None) and self.show_dead_cb.isChecked():
                x = self.sim.X; y = self.sim.Y
                # color alive vs dead
                colors = np.where(alive_mask[:, None], [1.0, 0.4, 0.4, 0.9], [0.4, 0.4, 0.4, 0.4])
            else:
                x = self.sim.X[alive_mask]; y = self.sim.Y[alive_mask]
                colors = np.tile([1.0, 0.4, 0.4, 0.9], (len(x), 1))
        except Exception:
            x = self.sim.X[alive_mask]; y = self.sim.Y[alive_mask]
            colors = np.tile([1.0, 0.4, 0.4, 0.9], (len(x), 1))

        pts = np.column_stack([x, y, np.zeros_like(x)])
        scatter = gl.GLScatterPlotItem(pos=pts, color=colors, size=6)
        try:
            if hasattr(self, 'gl_agent_scatter') and self.gl_agent_scatter is not None:
                self.gl_view.removeItem(self.gl_agent_scatter)
        except Exception:
            pass
        self.gl_agent_scatter = scatter
        self.gl_view.addItem(self.gl_agent_scatter)

        # Direction indicators (basic implementation)
        try:
            if getattr(self, 'show_direction_cb', None) and self.show_direction_cb.isChecked() and hasattr(self.sim, 'heading'):
                if hasattr(self, 'gl_direction_lines'):
                    for item in getattr(self, 'gl_direction_lines', []):
                        try:
                            self.gl_view.removeItem(item)
                        except Exception:
                            pass
                self.gl_direction_lines = []
                if getattr(self, 'show_dead_cb', None) and self.show_dead_cb.isChecked():
                    headings = self.sim.heading
                    pos_x, pos_y = self.sim.X, self.sim.Y
                else:
                    headings = self.sim.heading[alive_mask]
                    pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]
                arrow_length = 5.0
                end_x = pos_x - arrow_length * np.cos(headings)
                end_y = pos_y - arrow_length * np.sin(headings)
                for i in range(len(pos_x)):
                    seg_pts = np.array([[pos_x[i], pos_y[i], 0], [end_x[i], end_y[i], 0]])
                    seg = gl.GLLinePlotItem(pos=seg_pts, color=(1.0, 0.78, 0.39, 0.6), width=1.5, antialias=True, mode='lines')
                    self.gl_view.addItem(seg)
                    self.gl_direction_lines.append(seg)
        except Exception:
            pass
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
        try:
            if self.paused:
                self.play_btn.setEnabled(True)
                self.pause_btn.setText('Pause')
            else:
                self.play_btn.setEnabled(False)
                self.pause_btn.setText('Running')
        except Exception:
            pass

    def update_metrics_panel(self, metrics):
        """Update metrics panel labels (ported from original viewer)."""
        def fmt_metric(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val) if val is not None else "--"
        try:
            self.mean_speed_label.setText(f"Mean Speed: {fmt_metric(metrics.get('mean_speed', None))}")
            self.max_speed_label.setText(f"Max Speed: {fmt_metric(metrics.get('max_speed', None))}")
            self.mean_energy_label.setText(f"Mean Energy: {fmt_metric(metrics.get('mean_energy', None))}")
            self.min_energy_label.setText(f"Min Energy: {fmt_metric(metrics.get('min_energy', None))}")
            upstream_val = metrics.get('upstream_progress', metrics.get('mean_upstream_progress', '--'))
            if isinstance(upstream_val, (int, float)):
                self.upstream_progress_label.setText(f"Upstream Progress: {upstream_val:.2f}")
            else:
                self.upstream_progress_label.setText(f"Upstream Progress: --")
            self.mean_centerline_label.setText(f"Mean Centerline: {fmt_metric(metrics.get('mean_centerline', None))}")
            self.mean_passage_delay_label.setText(f"Mean Passage Delay: {fmt_metric(metrics.get('mean_passage_delay', None))}")
            if 'passage_success_rate' in metrics:
                self.passage_success_rate_label.setText(f"Passage Success: {metrics['passage_success_rate']:.1%}")
            elif 'success_rate' in metrics:
                self.passage_success_rate_label.setText(f"Passage Success: {metrics['success_rate']:.1%}")
            self.mean_nn_dist_label.setText(f"Mean NN Dist: {fmt_metric(metrics.get('mean_nn_dist', None))}")
            self.polarization_label.setText(f"Polarization: {fmt_metric(metrics.get('polarization', None))}")

            # accumulate per-timestep values into episode accumulators if checkboxes enabled
            for m, cb in getattr(self, 'track_metric_cbs', {}).items():
                try:
                    if cb.isChecked() and m in metrics:
                        if not hasattr(self, 'episode_metric_accumulators'):
                            self.episode_metric_accumulators = {}
                        if m not in self.episode_metric_accumulators:
                            self.episode_metric_accumulators[m] = []
                        self.episode_metric_accumulators[m].append(float(metrics[m]))
                except Exception:
                    pass
        except Exception:
            pass

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
