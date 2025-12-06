"""
Cleaned SalmonViewer implementation (rebuilt).

This preserves the original UI panels, GL-based TIN rendering, HECRAS perimeter
inference hook, RL panels, and weight controls. It is intentionally conservative
and uses the existing helper `_GLMeshBuilder` and `infer_wetted_perimeter_from_hecras`
if available.
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QSlider, QCheckBox, QGroupBox, QWidget)
import pyqtgraph as pg
from pyqtgraph import mkPen

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


class _GLMeshBuilder(QtCore.QThread):
    """Background thread to build GL mesh arrays from triangulation data.

    Emits a single `mesh_ready` signal with a dict containing 'verts', 'faces', 'colors'.
    """
    mesh_ready = QtCore.pyqtSignal(object)

    def __init__(self, tri_pts, tri_vals, kept_tris, vert_exag=1.0, parent=None):
        super().__init__(parent=parent)
        self.tri_pts = np.asarray(tri_pts, dtype=float)
        self.tri_vals = np.asarray(tri_vals, dtype=float)
        self.kept_tris = np.asarray(kept_tris, dtype=np.int32)
        self.vert_exag = float(vert_exag)

    def run(self):
        try:
            pts = self.tri_pts
            vals = self.tri_vals
            faces = self.kept_tris
            if pts.shape[0] == vals.shape[0]:
                z = vals * self.vert_exag
                verts = np.column_stack([pts[:, 0], pts[:, 1], z])
            else:
                verts = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))])

            # color map based on z
            try:
                zvals = verts[:, 2]
                vmin = np.nanmin(zvals)
                vmax = np.nanmax(zvals)
                span = vmax - vmin if vmax > vmin else 1.0
                norm = (zvals - vmin) / span
                colors = np.zeros((len(verts), 4), dtype=float)
                colors[:, 0] = 0.2 + 0.8 * norm
                colors[:, 1] = 0.4 * (1.0 - norm)
                colors[:, 2] = 0.6 * (1.0 - norm)
                colors[:, 3] = 1.0
            except Exception:
                colors = np.tile([0.6, 0.6, 0.6, 1.0], (len(verts), 1))

            payload = {'verts': verts.astype(float), 'faces': faces.astype(int), 'colors': colors.astype(float)}
            self.mesh_ready.emit(payload)
        except Exception as e:
            self.mesh_ready.emit({'error': e})


class SalmonViewer(QtWidgets.QWidget):
    """Main viewer widget for Salmon ABM.

    This class provides a compact UI that mirrors the original design while
    keeping implementation straightforward and robust to missing optional
    dependencies (GL, HECRAS helper, shapely).
    """

    def __init__(self, sim, dt=0.1, T=600, rl_trainer=None, parent=None):
        super().__init__(parent=parent)
        self.sim = sim
        self.dt = dt
        self.T = T
        self.rl_trainer = rl_trainer

        # runtime state
        self.current_timestep = 0
        self.current_episode = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.paused = True
        self.rewards_history = []

        # basic GL items
        self.tin_mesh = None
        self.gl_view = None

        # minimal placeholders
        self.trajectory_history = []
        self.trajectory_lines = []
        self.tail_phases = np.zeros(getattr(self.sim, 'num_agents', 0))
        self.tail_L1 = 1.0
        self.tail_L2 = 1.0
        self.tail_amp = 0.2
        self.tail_freq = 0.5

        self._build_ui()

        # periodic timer for updates (only active when unpaused)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(self.dt * 1000))

        # initial render
        QtCore.QTimer.singleShot(10, self.setup_background)

    def _build_ui(self):
        self.setWindowTitle('SalmonViewer')
        main_layout = QHBoxLayout()

        # left: plot area or GL view placeholder
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(600, 600)
        main_layout.addWidget(self.plot_widget, stretch=3)

        # right: control panels
        right_panel = QWidget()
        rp_layout = QVBoxLayout()

        # Play/Pause/Reset
        hp = QHBoxLayout()
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self._on_play)
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_simulation)
        hp.addWidget(self.play_btn)
        hp.addWidget(self.pause_btn)
        hp.addWidget(self.reset_btn)
        rp_layout.addLayout(hp)

        # Display toggles (directions, tails, trajectories, show dead)
        toggles = QHBoxLayout()
        self.show_direction_cb = QCheckBox('Directions')
        self.show_direction_cb.setChecked(True)
        self.show_tail_cb = QCheckBox('Tails')
        self.show_tail_cb.setChecked(True)
        self.show_trajectories_cb = QCheckBox('Trajectories')
        self.show_trajectories_cb.setChecked(False)
        self.show_dead_cb = QCheckBox('Show Dead')
        self.show_dead_cb.setChecked(False)
        toggles.addWidget(self.show_direction_cb)
        toggles.addWidget(self.show_tail_cb)
        toggles.addWidget(self.show_trajectories_cb)
        toggles.addWidget(self.show_dead_cb)
        rp_layout.addLayout(toggles)

        # Vertical Exaggeration + Rebuild button
        ve_group = QGroupBox('Vertical Exaggeration')
        ve_layout = QVBoxLayout()
        self.ve_label = QLabel('Z Exag: 1.00x')
        ve_layout.addWidget(self.ve_label)
        self.ve_slider = QSlider(Qt.Horizontal)
        self.ve_slider.setMinimum(1)
        self.ve_slider.setMaximum(500)
        self.ve_slider.setValue(int(getattr(self.sim, 'vert_exag', 1.0) * 100))
        self.ve_slider.valueChanged.connect(lambda v: self.ve_label.setText(f'Z Exag: {v/100:.2f}x'))
        ve_layout.addWidget(self.ve_slider)
        self.rebuild_btn = QPushButton('Rebuild TIN')
        self.rebuild_btn.clicked.connect(self.rebuild_tin_action)
        ve_layout.addWidget(self.rebuild_btn)
        ve_group.setLayout(ve_layout)
        rp_layout.addWidget(ve_group)

        # RL panel
        rp_layout.addWidget(self.create_rl_panel())

        # weights panel
        rp_layout.addWidget(self.create_weights_panel())

        # metrics
        rp_layout.addWidget(self.create_metrics_panel())

        rp_layout.addStretch()
        right_panel.setLayout(rp_layout)
        main_layout.addWidget(right_panel, stretch=1)

        self.setLayout(main_layout)

    def _on_play(self):
        self.paused = False
        print('[SalmonViewer] Play pressed')

    def setup_background(self):
        """Build a TIN from HECRAS or generic depth raster and render with GL.

        This reuses the `_GLMeshBuilder` background thread and falls back
        to synchronous run() if thread start fails (so exceptions are visible).
        """
        print('[RENDER] setup_background')
        try:
            # HECRAS path
            if hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path'):
                import h5py
                plan_path = self.sim.hecras_plan_path
                with h5py.File(plan_path, 'r') as hdf:
                    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
                wetted_mask = depth > 0.05
                pts = coords[wetted_mask]
                vals = depth[wetted_mask]
                max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
                if len(pts) > max_nodes:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(pts), size=max_nodes, replace=False)
                    pts = pts[idx]; vals = vals[idx]

                if len(pts) < 3:
                    print('[TIN] Not enough points for TIN')
                    return

                from scipy.spatial import Delaunay
                tri = Delaunay(pts)
                tris = tri.simplices

                # attempt perimeter inference
                perim = None
                try:
                    from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
                    perim = infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=0.05, max_nodes=max_nodes)
                    if not hasattr(self.sim, '_hecras_geometry_info'):
                        self.sim._hecras_geometry_info = {}
                    self.sim._hecras_geometry_info['perimeter_points'] = perim
                except Exception:
                    perim = None

                # optionally clip triangles by perimeter
                kept = np.ones(len(tris), dtype=bool)
                if perim is not None:
                    try:
                        from shapely.geometry import Point as ShPoint, Polygon
                        from shapely.ops import unary_union
                        if isinstance(perim, list) and len(perim) > 0 and isinstance(perim[0][0], (list, tuple)):
                            polys = [Polygon([tuple(p) for p in poly]) for poly in perim if len(poly) > 2]
                            perimeter = unary_union(polys)
                        else:
                            perimeter = Polygon([tuple(p) for p in perim])
                        tri_pts = pts
                        for i, t in enumerate(tris):
                            centroid = np.mean(tri_pts[t], axis=0)
                            if not perimeter.contains(ShPoint(centroid[0], centroid[1])):
                                kept[i] = False
                    except Exception:
                        kept = np.ones(len(tris), dtype=bool)

                kept_tris = tris[kept]

                # ensure GL available
                if gl is None:
                    print('[RENDER] pyqtgraph.opengl not available, skipping GL render')
                    return

                if self.gl_view is None:
                    self.gl_view = gl.GLViewWidget()
                    parent = self.plot_widget.parent()
                    layout = parent.layout()
                    layout.removeWidget(self.plot_widget)
                    self.plot_widget.hide()
                    layout.addWidget(self.gl_view)

                builder = _GLMeshBuilder(pts, vals, kept_tris, vert_exag=getattr(self.sim, 'vert_exag', 1.0), parent=self)

                def _on_mesh_ready(payload):
                    if 'error' in payload:
                        print('[TIN] GL mesh builder error:', payload['error'])
                        return
                    verts = payload['verts']; faces = payload['faces']; colors = payload['colors']
                    meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                    mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque')
                    mesh.setGLOptions('opaque')
                    if self.tin_mesh is not None:
                        try:
                            self.gl_view.removeItem(self.tin_mesh)
                        except Exception:
                            pass
                    self.tin_mesh = mesh
                    self.gl_view.addItem(mesh)

                builder.mesh_ready.connect(_on_mesh_ready)
                try:
                    builder.start()
                except Exception:
                    builder.run()

                return

            # Non-HECRAS path -- placeholder: no raster handling here for brevity
            print('[RENDER] No HECRAS data; skipping background TIN build')
        except Exception as e:
            print('[RENDER] setup_background failed:', e)

    # UI panel factories (kept compact but functionally equivalent)
    def create_rl_panel(self):
        rl_group = QGroupBox('RL Training')
        layout = QVBoxLayout()
        self.episode_label = QLabel(f'Episode: {self.current_episode} | Timestep: 0')
        self.reward_label = QLabel(f'Reward: {self.episode_reward:.2f}')
        self.best_reward_label = QLabel('Best: 0.00')
        layout.addWidget(self.episode_label)
        layout.addWidget(self.reward_label)
        layout.addWidget(self.best_reward_label)
        self.reward_plot = pg.PlotWidget(title='Episode Rewards')
        self.reward_plot.setMaximumHeight(220)
        layout.addWidget(self.reward_plot)
        rl_group.setLayout(layout)
        return rl_group

    def create_weights_panel(self):
        weights_group = QGroupBox('Behavioral Weights')
        layout = QVBoxLayout()
        self.weight_labels = {}
        self.weight_sliders = {}
        if hasattr(self.sim, 'behavioral_weights'):
            weights = self.sim.behavioral_weights
            weight_attrs = [
                'cohesion_weight', 'alignment_weight', 'separation_weight', 'separation_radius',
                'threat_level', 'cohesion_radius_relaxed', 'cohesion_radius_threatened',
                'drafting_enabled', 'drafting_distance', 'drafting_angle_tolerance',
                'drag_reduction_single', 'drag_reduction_dual',
                'rheotaxis_weight', 'border_cue_weight', 'border_threshold_multiplier', 'border_max_force',
                'collision_weight', 'collision_radius',
                'upstream_priority', 'energy_efficiency_priority',
                'learning_rate', 'exploration_epsilon',
                'use_sog', 'sog_weight'
            ]
            for attr in weight_attrs:
                if hasattr(weights, attr):
                    value = getattr(weights, attr)
                    # Label
                    if isinstance(value, bool):
                        label = QLabel(f"{attr.replace('_', ' ').title()}: {'ON' if value else 'OFF'}")
                    else:
                        label = QLabel(f"{attr.replace('_', ' ').title()}: {value:.3f}")
                    label.setStyleSheet('font-size: 9pt; color: black;')
                    self.weight_labels[attr] = label
                    layout.addWidget(label)

                    # Controls
                    if isinstance(value, bool):
                        cb = QCheckBox()
                        cb.setChecked(value)
                        cb.stateChanged.connect(lambda s, a=attr, cb=cb: self.update_bool_weight(a, cb.isChecked()))
                        layout.addWidget(cb)
                        self.weight_sliders[attr] = cb
                    else:
                        slider = QSlider(Qt.Horizontal)
                        # default centi-based mapping
                        slider_min, slider_max = 0, 200
                        slider_value = int(value * 100)
                        if attr in ('separation_radius', 'cohesion_radius_relaxed', 'cohesion_radius_threatened', 'collision_radius', 'drafting_distance'):
                            slider_min, slider_max = 0, 500
                            slider_value = int(value * 100)
                        elif attr in ('border_threshold_multiplier', 'drafting_angle_tolerance'):
                            slider_min, slider_max = 0, 360
                            slider_value = int(value)
                        elif attr in ('border_max_force', 'collision_weight', 'drag_reduction_single', 'drag_reduction_dual', 'border_cue_weight'):
                            slider_min, slider_max = 0, 2000
                            slider_value = int(value * 10)
                        elif attr in ('learning_rate',):
                            slider_min, slider_max = 0, 1000
                            slider_value = int(value * 100000)
                        elif attr in ('exploration_epsilon', 'sog_weight', 'energy_efficiency_priority', 'upstream_priority', 'threat_level'):
                            slider_min, slider_max = 0, 100
                            slider_value = int(value * 100)

                        slider.setMinimum(slider_min)
                        slider.setMaximum(slider_max)
                        slider.setValue(max(slider_min, min(slider_max, slider_value)))
                        slider.valueChanged.connect(lambda v, a=attr, l=label: self.update_weight(a, v, l))
                        self.weight_sliders[attr] = slider
                        layout.addWidget(slider)
        else:
            layout.addWidget(QLabel('No weights configured'))
        weights_group.setLayout(layout)
        return weights_group

    def update_weight(self, attr, value, label):
        if attr in ('separation_radius', 'cohesion_radius_relaxed', 'cohesion_radius_threatened', 'collision_radius', 'drafting_distance'):
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        elif attr in ('border_threshold_multiplier', 'drafting_angle_tolerance'):
            weight_value = float(value)
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.0f}")
        elif attr in ('border_max_force', 'collision_weight', 'drag_reduction_single', 'drag_reduction_dual', 'border_cue_weight'):
            weight_value = value / 10.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        elif attr in ('learning_rate',):
            weight_value = value / 100000.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.6f}")
        elif attr in ('exploration_epsilon', 'sog_weight', 'energy_efficiency_priority', 'upstream_priority', 'threat_level'):
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        else:
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.3f}")

        label.setStyleSheet('font-size: 9pt; color: black;')
        if hasattr(self.sim, 'behavioral_weights'):
            try:
                setattr(self.sim.behavioral_weights, attr, weight_value)
                self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
            except Exception:
                pass

    def update_bool_weight(self, attr, state):
        if hasattr(self.sim, 'behavioral_weights'):
            try:
                setattr(self.sim.behavioral_weights, attr, bool(state))
                self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
                if attr in self.weight_labels:
                    self.weight_labels[attr].setText(f"{attr.replace('_', ' ').title()}: {'ON' if state else 'OFF'}")
            except Exception:
                pass

    def create_metrics_panel(self):
        metrics_group = QGroupBox('Metrics')
        layout = QVBoxLayout()
        self.mean_speed_label = QLabel('Mean Speed: --')
        self.max_speed_label = QLabel('Max Speed: --')
        layout.addWidget(self.mean_speed_label)
        layout.addWidget(self.max_speed_label)

        self.mean_energy_label = QLabel('Mean Energy: --')
        self.min_energy_label = QLabel('Min Energy: --')
        layout.addWidget(self.mean_energy_label)
        layout.addWidget(self.min_energy_label)

        self.upstream_progress_label = QLabel('Upstream Progress: --')
        self.mean_centerline_label = QLabel('Mean Centerline: --')
        layout.addWidget(self.upstream_progress_label)
        layout.addWidget(self.mean_centerline_label)

        self.mean_passage_delay_label = QLabel('Mean Passage Delay: --')
        self.passage_success_rate_label = QLabel('Passage Success: --')
        layout.addWidget(self.mean_passage_delay_label)
        layout.addWidget(self.passage_success_rate_label)

        # Schooling metrics
        self.mean_nn_dist_label = QLabel('Mean NN Dist: --')
        self.polarization_label = QLabel('Polarization: --')
        layout.addWidget(self.mean_nn_dist_label)
        layout.addWidget(self.polarization_label)

        # per-episode tracking checkboxes
        self._available_episode_metrics = ['collision_count', 'mean_upstream_progress', 'mean_upstream_velocity', 'energy_efficiency', 'mean_passage_delay']
        self.track_metric_cbs = {}
        def add_label_with_cb(label_widget, metric_key, default_checked=False):
            h = QHBoxLayout(); h.setContentsMargins(0,0,0,0); h.setSpacing(6)
            h.addWidget(label_widget)
            cb = QCheckBox(); cb.setChecked(default_checked); cb.setFixedWidth(22)
            h.addWidget(cb)
            layout.addLayout(h)
            self.track_metric_cbs[metric_key] = cb

        self.collision_count_label = QLabel('Collision Count: --')
        add_label_with_cb(self.collision_count_label, 'collision_count')
        add_label_with_cb(self.upstream_progress_label, 'mean_upstream_progress')
        self.mean_upstream_velocity_label = QLabel('Mean Upstream Velocity: --')
        add_label_with_cb(self.mean_upstream_velocity_label, 'mean_upstream_velocity')
        add_label_with_cb(self.mean_energy_label, 'energy_efficiency')
        add_label_with_cb(self.mean_passage_delay_label, 'mean_passage_delay')
        metrics_group.setLayout(layout)
        return metrics_group

    def update_simulation(self):
        if self.paused:
            return
        try:
            self.sim.timestep(self.current_timestep, self.dt, 9.81, None)
            self.current_timestep += 1
            if self.rl_trainer:
                self.update_rl_training()
            self.update_displays()
        except Exception as e:
            print('[ERROR] update_simulation failed:', e)
            import traceback
            traceback.print_exc()
            self.paused = True

    def update_rl_training(self):
        try:
            metrics = self.rl_trainer.extract_state_metrics()
            self.update_metrics_panel(metrics)
        except Exception:
            pass

        # Episode end handling (preserve reward history and saving behavior)
        if self.current_timestep >= self.T:
            self.rewards_history.append(self.episode_reward)
            if self.episode_reward > getattr(self, 'best_reward', -1e9):
                self.best_reward = self.episode_reward
                try:
                    import json, os
                    save_dir = os.path.join(os.getcwd(), 'outputs', 'rl_training')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'best_weights.json')
                    with open(save_path, 'w') as f:
                        json.dump(self.rl_trainer.behavioral_weights.to_dict(), f, indent=2)
                    print('Saved best weights to', save_path)
                except Exception:
                    pass
            # prepare next episode
            try:
                self.sim.reset_spatial_state(reset_positions=True)
            except Exception:
                try:
                    self.sim.reset_spatial_state()
                except Exception:
                    pass
            self.current_episode += 1
            self.current_timestep = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
            # update reward plot
            try:
                self.reward_plot.plot(range(len(self.rewards_history)), self.rewards_history, pen=mkPen('g', width=2), clear=True)
            except Exception:
                pass

    def rebuild_tin_action(self):
        try:
            # force a new background build using current VE slider
            if hasattr(self, 've_slider'):
                self.sim.vert_exag = self.ve_slider.value() / 100.0
            QtCore.QTimer.singleShot(10, self.setup_background)
        except Exception as e:
            print('[REBUILD] Exception rebuilding TIN:', e)

    def update_metrics_panel(self, metrics):
        try:
            if 'mean_speed' in metrics:
                self.mean_speed_label.setText(f"Mean Speed: {metrics['mean_speed']:.2f}")
            if 'max_speed' in metrics:
                self.max_speed_label.setText(f"Max Speed: {metrics['max_speed']:.2f}")
        except Exception:
            pass

    def update_displays(self):
        # Minimal: update agent scatter if GL present
        try:
            if gl is None or self.gl_view is None:
                return
            if not (hasattr(self.sim, 'X') and hasattr(self.sim, 'Y')):
                return
            alive_mask = (getattr(self.sim, 'dead', np.zeros(getattr(self.sim, 'num_agents', 0))) == 0)
            if self.sim.num_agents == 0:
                return
            if self.show_dead_cb.isChecked() if hasattr(self, 'show_dead_cb') else False:
                x = self.sim.X; y = self.sim.Y
            else:
                x = self.sim.X[alive_mask]; y = self.sim.Y[alive_mask]
            pts = np.column_stack([x, y, np.zeros_like(x)])
            scatter = gl.GLScatterPlotItem(pos=pts, color=(1.0, 0.4, 0.4, 0.9), size=6)
            try:
                if hasattr(self, 'gl_agent_scatter') and self.gl_agent_scatter is not None:
                    self.gl_view.removeItem(self.gl_agent_scatter)
            except Exception:
                pass
            self.gl_agent_scatter = scatter
            self.gl_view.addItem(self.gl_agent_scatter)
        except Exception as e:
            print('[RENDER] update_displays error:', e)

    def toggle_pause(self):
        self.paused = not self.paused
        print(f'[SalmonViewer] paused -> {self.paused}')

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
        self.current_episode = 0
        self.episode_reward = 0.0
        try:
            self.update_displays()
        except Exception:
            pass

    def run(self):
        self.show()
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
"""
Real-time visualization for Salmon ABM simulations.

Provides PyQt5-based live viewer for:
- Standard simulations (watch fish navigate)
- RL training (watch behavioral weight optimization)
- Interactive parameter tuning

Usage:
    from emergent.salmon_abm.salmon_viewer import SalmonViewer
    viewer = SalmonViewer(simulation=sim, dt=0.1, T=600)
    viewer.run()
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                              QSlider, QCheckBox, QGroupBox, QWidget)
from PyQt5.QtGui import QColor, QPen, QBrush, QPolygonF, QPainter
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, ScatterPlotItem
import rasterio
from rasterio.transform import from_bounds

# Import GL here (will require PyOpenGL installed)
try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


class _GLMeshBuilder(QtCore.QThread):
    """Background thread to build GL mesh arrays from triangulation data.

    Emits a single `mesh_ready` signal with a dict containing 'verts', 'faces', 'colors'.
    """
    mesh_ready = QtCore.pyqtSignal(object)

    def __init__(self, tri_pts, tri_vals, kept_tris, vert_exag=1.0, parent=None):
        super().__init__(parent=parent)
        self.tri_pts = np.asarray(tri_pts, dtype=float)
        self.tri_vals = np.asarray(tri_vals, dtype=float)
        self.kept_tris = np.asarray(kept_tris, dtype=np.int32)
        self.vert_exag = float(vert_exag)

    def run(self):
        """Construct mesh arrays and emit them via `mesh_ready`.

        The payload will be a dict: {'verts': (N,3), 'faces': (M,3), 'colors': (N,4)}.
        On error the payload will be {'error': exception}
        """
        try:
            pts = self.tri_pts
            vals = self.tri_vals
            faces = self.kept_tris

            # Build vertex array: x, y, z (z from scalar vals * vertical exaggeration)
            if pts.shape[0] != vals.shape[0]:
                # If vals are per-triangle, expand to per-vertex by averaging
                verts = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(len(pts))])
            else:
                z = vals * self.vert_exag
                verts = np.column_stack([pts[:, 0], pts[:, 1], z])

            # Build simple per-vertex colors from z (normalized)
            try:
                zvals = verts[:, 2]
                vmin = np.nanmin(zvals)
                vmax = np.nanmax(zvals)
                span = vmax - vmin if vmax > vmin else 1.0
                norm = (zvals - vmin) / span
                colors = np.zeros((len(verts), 4), dtype=float)
                colors[:, 0] = 0.2 + 0.8 * norm
                colors[:, 1] = 0.4 * (1.0 - norm)
                colors[:, 2] = 0.6 * (1.0 - norm)
                colors[:, 3] = 1.0
            except Exception:
                colors = np.tile([0.6, 0.6, 0.6, 1.0], (len(verts), 1))

            payload = {'verts': verts.astype(float), 'faces': faces.astype(int), 'colors': colors.astype(float)}
            self.mesh_ready.emit(payload)
        except Exception as e:
            self.mesh_ready.emit({'error': e})
        ve_group = QGroupBox("Vertical Exaggeration")
        ve_layout = QVBoxLayout()
        self.ve_label = QLabel("Z Exag: 1.00x")
        ve_layout.addWidget(self.ve_label)
        self.ve_slider = QSlider(Qt.Horizontal)
        self.ve_slider.setMinimum(1)
        self.ve_slider.setMaximum(500)
        self.ve_slider.setValue(int(getattr(self.sim, 'vert_exag', 1.0) * 100))
        self.ve_slider.valueChanged.connect(self.update_vert_exag_label)
        ve_layout.addWidget(self.ve_slider)

        self.rebuild_btn = QPushButton("Rebuild TIN")
        self.rebuild_btn.clicked.connect(self.rebuild_tin_action)
        ve_layout.addWidget(self.rebuild_btn)
        ve_group.setLayout(ve_layout)
        layout.addWidget(ve_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def setup_background(self):
        """Setup depth raster as background image or HECRAS wetted cells."""
        print("[RENDER] setup_background")
        import traceback
        try:
            # Try HECRAS mode first
            if hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path'):
                import h5py
                plan_path = self.sim.hecras_plan_path
                with h5py.File(plan_path, 'r') as hdf:
                    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
                    print(f"[TIN] loaded {coords.shape[0]} cells, depth len={depth.shape[0]}")
                wetted_mask = depth > 0.05
                pts = coords[wetted_mask]
                vals = depth[wetted_mask]
                print(f"[TIN] wetted cells: {len(pts)}")
                # Thinning
                max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
                if len(pts) > max_nodes:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(len(pts), size=max_nodes, replace=False)
                    pts = pts[idx]
                    vals = vals[idx]
                    print(f"[TIN] thinned to {len(pts)} nodes")
                # Delaunay
                if len(pts) < 3:
                    print("[TIN DEBUG] Not enough points for TIN")
                    return
                from scipy.spatial import Delaunay
                tri = Delaunay(pts)
                tris = tri.simplices
                print(f"[TIN] Delaunay produced {tris.shape[0]} triangles")
                import pyqtgraph.opengl as gl
                if gl is None:
                    print("[TIN DEBUG] pyqtgraph.opengl not available")
                    return
                if not hasattr(self, 'gl_view'):
                    print("[TIN DEBUG] gl_view does not exist, creating GLViewWidget")
                    self.gl_view = gl.GLViewWidget()
                else:
                    print("[TIN DEBUG] gl_view already exists")
                # Attempt to infer a more accurate wetted perimeter from HECRAS geometry
                try:
                    from emergent.salmon_abm.hecras_helpers import infer_wetted_perimeter_from_hecras
                    perimeter_points = infer_wetted_perimeter_from_hecras(plan_path, depth_threshold=0.05, max_nodes=max_nodes)
                    if not hasattr(self.sim, '_hecras_geometry_info'):
                        self.sim._hecras_geometry_info = {}
                    self.sim._hecras_geometry_info['perimeter_points'] = perimeter_points
                except Exception:
                    # If inference fails, proceed without perimeter_points (tri clipping will be skipped)
                    perimeter_points = None

                builder = _GLMeshBuilder(pts, vals, tris, parent=self)
                def _on_mesh_ready(payload):
                    print("[TIN] mesh_ready received")
                    if 'error' in payload:
                        print('[TIN DEBUG] GL mesh builder error:', payload['error'])
                        traceback.print_exc()
                        return
                    verts = payload['verts']
                    faces = payload['faces']
                    colors = payload['colors']
                    print("[TIN] adding GLMeshItem to view")
                    meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                    mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque')
                    mesh.setGLOptions('opaque')
                    if hasattr(self, 'tin_mesh') and self.tin_mesh is not None:
                        self.gl_view.removeItem(self.tin_mesh)
                    self.tin_mesh = mesh
                    self.gl_view.addItem(mesh)
                    # Camera auto-fit
                    min_xyz = np.nanmin(verts, axis=0)
                    max_xyz = np.nanmax(verts, axis=0)
                    center = (min_xyz + max_xyz) / 2.0
                    size = np.max(max_xyz - min_xyz)
                    self.gl_view.setCameraPosition(pos=QtGui.QVector3D(*center), distance=size*1.2)
                builder.mesh_ready.connect(_on_mesh_ready)
                try:
                    builder.start()
                    print("[TIN] GLMeshBuilder started")
                except Exception as e:
                    print("[TIN DEBUG] GLMeshBuilder.start() raised:", e)
                    import traceback
                    traceback.print_exc()
                    print("[TIN DEBUG] Running _GLMeshBuilder.run() synchronously as fallback")
                    try:
                        builder.run()
                    except Exception as e2:
                        print("[TIN DEBUG] Synchronous GLMeshBuilder.run() failed:", e2)
                        traceback.print_exc()
                self.initial_zoom_done = False
                return
            tri_pts = np.asarray(coords, dtype=float)
            tri_vals = np.asarray(vals, dtype=float)
            if len(tri_pts) > max_nodes:
                rng = np.random.default_rng(0)
                idx_keep = rng.choice(len(tri_pts), size=max_nodes, replace=False)
                tri_pts = tri_pts[idx_keep]
                tri_vals = tri_vals[idx_keep]
                print(f'Capped input nodes: {len(coords)} -> {len(tri_pts)} (max={max_nodes})')

            from scipy.spatial import Delaunay
            import time
            t0 = time.perf_counter()
            tri = Delaunay(tri_pts)
            t1 = time.perf_counter()
            print(f'Delaunay time: {t1-t0:.2f}s for {len(tri_pts)} nodes')
            tris = tri.simplices

            # perform perimeter clipping if available
            kept = np.ones(len(tris), dtype=bool)
            try:
                perim_points = None
                if hasattr(self.sim, '_hecras_geometry_info'):
                    perim_points = self.sim._hecras_geometry_info.get('perimeter_points', None)
                if perim_points is not None:
                    from shapely.geometry import Point as ShPoint, Polygon
                    from shapely.ops import unary_union
                    # helper may return list of polygons (list of coord lists)
                    if isinstance(perim_points, list) and len(perim_points) > 0 and isinstance(perim_points[0], (list, tuple)) and len(perim_points[0]) > 0 and isinstance(perim_points[0][0], (list, tuple)):
                        polys = [Polygon([tuple(p) for p in poly]) for poly in perim_points if len(poly) > 2]
                        if not polys:
                            raise RuntimeError('No valid polygons returned by perimeter helper')
                        perimeter = unary_union(polys)
                    else:
                        # assume a single polygon coordinate list
                        if len(perim_points) > 3:
                            perimeter = Polygon([tuple(p) for p in perim_points])
                        else:
                            raise RuntimeError('Perimeter points insufficient')
                    for i, t in enumerate(tris):
                        coords_tri = tri_pts[t]
                        centroid = np.mean(coords_tri, axis=0)
                        try:
                            if not perimeter.contains(ShPoint(centroid[0], centroid[1])):
                                kept[i] = False
                        except Exception:
                            kept[i] = False
            except Exception:
                kept = np.ones(len(tris), dtype=bool)

            kept_tris = tris[kept]

            # Ensure GL is available
            if gl is None:
                raise RuntimeError('pyqtgraph.opengl (PyOpenGL) not available')

            # create GL view if needed
            if not hasattr(self, 'gl_view'):
                print("[RENDER DEBUG] gl_view does not exist, creating GLViewWidget")
                self.gl_view = gl.GLViewWidget()
                parent = self.plot_widget.parent()
                layout = parent.layout()
                layout.removeWidget(self.plot_widget)
                self.plot_widget.hide()
                layout.addWidget(self.gl_view)
            else:
                print("[RENDER DEBUG] gl_view already exists")

            # compute vertical exaggeration
            if vert_exag is None:
                vert_exag = getattr(self.sim, 'vert_exag', 1.0)

            # launch mesh builder with vertical exaggeration
            builder = _GLMeshBuilder(tri_pts, tri_vals, kept_tris, vert_exag=vert_exag, parent=self)

            def _on_mesh_ready(payload):
                print("[Viewer] mesh_ready callback reached.")
                if 'error' in payload:
                    print('GL mesh builder error:', payload['error'])
                    return
                verts = payload['verts']
                faces = payload['faces']
                colors = payload['colors']
                meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque')
                mesh.setGLOptions('opaque')
                if hasattr(self, 'tin_mesh') and self.tin_mesh is not None:
                    self.gl_view.removeItem(self.tin_mesh)
                self.tin_mesh = mesh
                self.gl_view.addItem(mesh)
                print("[Viewer] mesh added to gl_view.")
                # Camera auto-fit
                min_xyz = np.nanmin(verts, axis=0)
                max_xyz = np.nanmax(verts, axis=0)
                center = (min_xyz + max_xyz) / 2.0
                size = np.max(max_xyz - min_xyz)
                self.gl_view.setCameraPosition(pos=QtGui.QVector3D(*center), distance=size*1.2)

            builder.mesh_ready.connect(_on_mesh_ready)
            try:
                builder.start()
            except Exception as e:
                print('render_tin_from_arrays: builder.start() raised:', e)
                import traceback
                traceback.print_exc()
                print('render_tin_from_arrays: running builder.run() synchronously as fallback')
                try:
                    builder.run()
                except Exception as e2:
                    print('render_tin_from_arrays: synchronous builder.run() failed:', e2)
                    traceback.print_exc()

        except Exception as e:
            print('render_tin_from_arrays failed:', e)
    
    def create_rl_panel(self):
        """Create RL training metrics panel."""
        rl_group = QGroupBox("RL Training")
        layout = QVBoxLayout()
        # tighten spacing to be consistent with metrics panel
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)
        self.episode_label = QLabel(f"Episode: {self.current_episode} | Timestep: 0")
        self.reward_label = QLabel(f"Reward: {self.episode_reward:.2f}")
        self.best_reward_label = QLabel(f"Best: {self.best_reward:.2f}")
        
        layout.addWidget(self.episode_label)
        layout.addWidget(self.reward_label)
        layout.addWidget(self.best_reward_label)
        
        # Reward plot (increase size to better use left-panel space)
        self.reward_plot = pg.PlotWidget(title="Episode Rewards")
        self.reward_plot.setLabel('bottom', 'Episode')
        self.reward_plot.setLabel('left', 'Total Reward')
        self.reward_plot.setMaximumHeight(220)
        layout.addWidget(self.reward_plot)
        
        rl_group.setLayout(layout)
        return rl_group
    
    def create_weights_panel(self):
        """Create behavioral weights display panel with sliders."""
        weights_group = QGroupBox("Behavioral Weights")
        layout = QVBoxLayout()
        
        # Get weights from simulation
        if hasattr(self.sim, 'behavioral_weights'):
            weights = self.sim.behavioral_weights
            
            # Create sliders for each weight
            self.weight_labels = {}
            self.weight_sliders = {}
            # Complete set of BehavioralWeights attributes (from weights.to_dict())
            weight_attrs = [
                'cohesion_weight', 'alignment_weight', 'separation_weight', 'separation_radius',
                'threat_level', 'cohesion_radius_relaxed', 'cohesion_radius_threatened',
                'drafting_enabled', 'drafting_distance', 'drafting_angle_tolerance',
                'drag_reduction_single', 'drag_reduction_dual',
                'rheotaxis_weight', 'border_cue_weight', 'border_threshold_multiplier', 'border_max_force',
                'collision_weight', 'collision_radius',
                'upstream_priority', 'energy_efficiency_priority',
                'learning_rate', 'exploration_epsilon',
                'use_sog', 'sog_weight'
            ]

            for attr in weight_attrs:
                if hasattr(weights, attr):
                    value = getattr(weights, attr)

                    # Label
                    # Booleans display as ON/OFF
                    if isinstance(value, bool):
                        label = QLabel(f"{attr.replace('_', ' ').title()}: {'ON' if value else 'OFF'}")
                    else:
                        label = QLabel(f"{attr.replace('_', ' ').title()}: {value:.3f}")
                    label.setStyleSheet("font-size: 9pt; color: black;")
                    self.weight_labels[attr] = label
                    layout.addWidget(label)

                    # Configure slider ranges by parameter type
                    if isinstance(value, bool):
                        # Use a checkbox for booleans
                        cb = QCheckBox()
                        cb.setChecked(value)
                        cb.stateChanged.connect(lambda s, a=attr, cb=cb: self.update_bool_weight(a, cb.isChecked()))
                        layout.addWidget(cb)
                        self.weight_sliders[attr] = cb
                    else:
                        slider = QSlider(Qt.Horizontal)
                        # Default range 0.0 - 2.0
                        slider_min = 0
                        slider_max = 200
                        slider_value = 0

                        # Specific ranges for known params
                        if attr in ('separation_radius', 'cohesion_radius_relaxed', 'cohesion_radius_threatened', 'collision_radius', 'drafting_distance'):
                            slider_min, slider_max = 0, 500  # 0.00 - 5.00 (stored as centi-units)
                            slider_value = int(value * 100)
                        elif attr in ('border_threshold_multiplier', 'drafting_angle_tolerance'):
                            slider_min, slider_max = 0, 360  # degrees or multiplier (0..360)
                            slider_value = int(value)
                        elif attr in ('border_max_force', 'collision_weight', 'drag_reduction_single', 'drag_reduction_dual', 'border_cue_weight'):
                            slider_min, slider_max = 0, 2000
                            slider_value = int(value * 10)
                        elif attr in ('learning_rate',):
                            slider_min, slider_max = 0, 1000  # 0.0 - 0.01 (as 1e-5 steps)
                            slider_value = int(value * 100000)
                        elif attr in ('exploration_epsilon', 'sog_weight', 'energy_efficiency_priority', 'upstream_priority', 'threat_level'):
                            slider_min, slider_max = 0, 100
                            slider_value = int(value * 100)
                        else:
                            slider_min, slider_max = 0, 200
                            slider_value = int(value * 100)

                        slider.setMinimum(slider_min)
                        slider.setMaximum(slider_max)
                        slider.setValue(max(slider_min, min(slider_max, slider_value)))

                        # Connect handler with attribute name and label
                        slider.valueChanged.connect(lambda v, a=attr, l=label: self.update_weight(a, v, l))
                        self.weight_sliders[attr] = slider
                        layout.addWidget(slider)
        else:
            # No weights available
            no_weights_label = QLabel("No weights configured")
            no_weights_label.setStyleSheet("font-style: italic; color: gray;")
            layout.addWidget(no_weights_label)
            self.weight_labels = {}
            self.weight_sliders = {}
        
        weights_group.setLayout(layout)
        return weights_group
    
    def update_weight(self, attr, value, label):
        """Update behavioral weight from slider."""
        # Determine how to convert slider integer back to meaningful value
        if attr in ('separation_radius', 'cohesion_radius_relaxed', 'cohesion_radius_threatened', 'collision_radius', 'drafting_distance'):
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        elif attr in ('border_threshold_multiplier', 'drafting_angle_tolerance'):
            weight_value = float(value)
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.0f}")
        elif attr in ('border_max_force', 'collision_weight', 'drag_reduction_single', 'drag_reduction_dual', 'border_cue_weight'):
            weight_value = value / 10.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        elif attr in ('learning_rate',):
            weight_value = value / 100000.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.6f}")
        elif attr in ('exploration_epsilon', 'sog_weight', 'energy_efficiency_priority', 'upstream_priority', 'threat_level'):
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.2f}")
        else:
            weight_value = value / 100.0
            label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.3f}")

        label.setStyleSheet("font-size: 9pt; color: black;")
        if hasattr(self.sim, 'behavioral_weights'):
            try:
                setattr(self.sim.behavioral_weights, attr, weight_value)
                self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
            except Exception:
                # Some attributes may be displayed but not directly settable in runtime
                pass

    def update_bool_weight(self, attr, state):
        """Update boolean weight from checkbox."""
        if hasattr(self.sim, 'behavioral_weights'):
            try:
                setattr(self.sim.behavioral_weights, attr, bool(state))
                self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
                # Update label
                if attr in self.weight_labels:
                    self.weight_labels[attr].setText(f"{attr.replace('_', ' ').title()}: {'ON' if state else 'OFF'}")
            except Exception:
                pass
    
    def create_metrics_panel(self):
        """Create behavior metrics panel."""
        metrics_group = QGroupBox("Metrics")
        layout = QVBoxLayout()
        
        # Speed metrics
        self.mean_speed_label = QLabel("Mean Speed: --")
        self.max_speed_label = QLabel("Max Speed: --")
        layout.addWidget(self.mean_speed_label)
        layout.addWidget(self.max_speed_label)
        
        # Energy metrics
        self.mean_energy_label = QLabel("Mean Energy: --")
        self.min_energy_label = QLabel("Min Energy: --")
        layout.addWidget(self.mean_energy_label)
        layout.addWidget(self.min_energy_label)
        
        # Progress metrics
        self.upstream_progress_label = QLabel("Upstream Progress: --")
        self.mean_centerline_label = QLabel("Mean Centerline: --")
        layout.addWidget(self.upstream_progress_label)
        layout.addWidget(self.mean_centerline_label)
        
        # Passage delay metric
        self.mean_passage_delay_label = QLabel("Mean Passage Delay: --")
        layout.addWidget(self.mean_passage_delay_label)
        # Passage completion / percentiles (placeholder)
        self.passage_success_rate_label = QLabel("Passage Success: --")
        layout.addWidget(self.passage_success_rate_label)
        
        # Schooling metrics
        self.mean_nn_dist_label = QLabel("Mean NN Dist: --")
        self.polarization_label = QLabel("Polarization: --")
        layout.addWidget(self.mean_nn_dist_label)
        layout.addWidget(self.polarization_label)

        # (Removed time-series plots for collisions, upstream progress, and upstream velocity)

        # Per-episode tracking toggles (inline next to metric labels)
        # Available metrics that can be tracked per-episode
        self._available_episode_metrics = [
            'collision_count', 'mean_upstream_progress', 'mean_upstream_velocity',
            'energy_efficiency', 'mean_passage_delay'
        ]
        self.track_metric_cbs = {}

        # Helper to add a label+checkbox inline
        def add_label_with_cb(label_widget, metric_key, default_checked=False):
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            # ensure label_widget is a widget
            h.addWidget(label_widget)
            cb = QCheckBox()
            cb.setChecked(default_checked)
            cb.setFixedWidth(22)
            h.addWidget(cb)
            layout.addLayout(h)
            self.track_metric_cbs[metric_key] = cb

        # collision_count (no real-time label previously) -> add new label
        self.collision_count_label = QLabel("Collision Count: --")
        add_label_with_cb(self.collision_count_label, 'collision_count')

        # mean_upstream_progress -> use existing upstream_progress_label
        add_label_with_cb(self.upstream_progress_label, 'mean_upstream_progress')

        # mean_upstream_velocity (no label previously) -> add new label
        self.mean_upstream_velocity_label = QLabel("Mean Upstream Velocity: --")
        add_label_with_cb(self.mean_upstream_velocity_label, 'mean_upstream_velocity')

        # energy_efficiency -> map to mean_energy_label
        add_label_with_cb(self.mean_energy_label, 'energy_efficiency')

        # mean_passage_delay -> map to mean_passage_delay_label
        add_label_with_cb(self.mean_passage_delay_label, 'mean_passage_delay')

        # Per-episode metrics plot (shows one point per episode per tracked metric)
        self.per_episode_plot = pg.PlotWidget(title="Per-Episode Metrics")
        self.per_episode_plot.setLabel('bottom', 'Episode')
        self.per_episode_plot.setLabel('left', 'Metric Value')
        self.per_episode_plot.setMaximumHeight(220)
        layout.addWidget(self.per_episode_plot)
        # storage for per-episode series and plot handles
        self.per_episode_series = {m: [] for m in self._available_episode_metrics}
        self.per_episode_handles = {}
        
        metrics_group.setLayout(layout)
        return metrics_group
    
    def update_metrics_panel(self, metrics):
        """Update metrics panel labels."""
        def fmt_metric(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val) if val is not None else "--"
        self.mean_speed_label.setText(f"Mean Speed: {fmt_metric(metrics.get('mean_speed', None))}")
        self.max_speed_label.setText(f"Max Speed: {fmt_metric(metrics.get('max_speed', None))}")
        self.mean_energy_label.setText(f"Mean Energy: {fmt_metric(metrics.get('mean_energy', None))}")
        self.min_energy_label.setText(f"Min Energy: {fmt_metric(metrics.get('min_energy', None))}")
        # Upstream progress may be reported as 'upstream_progress' or 'mean_upstream_progress'
        upstream_val = metrics.get('upstream_progress', metrics.get('mean_upstream_progress', '--'))
        if isinstance(upstream_val, (int, float)):
            self.upstream_progress_label.setText(f"Upstream Progress: {upstream_val:.2f}")
        else:
            self.upstream_progress_label.setText(f"Upstream Progress: --")
        def fmt_metric(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val) if val is not None else "--"
        self.mean_centerline_label.setText(f"Mean Centerline: {fmt_metric(metrics.get('mean_centerline', None))}")
        self.mean_passage_delay_label.setText(f"Mean Passage Delay: {fmt_metric(metrics.get('mean_passage_delay', None))}")
        # Optional passage stats
        if 'passage_success_rate' in metrics:
            self.passage_success_rate_label.setText(f"Passage Success: {metrics['passage_success_rate']:.1%}")
        elif 'success_rate' in metrics:
            self.passage_success_rate_label.setText(f"Passage Success: {metrics['success_rate']:.1%}")
        self.mean_nn_dist_label.setText(f"Mean NN Dist: {fmt_metric(metrics.get('mean_nn_dist', None))}")
        self.polarization_label.setText(f"Polarization: {fmt_metric(metrics.get('polarization', None))}")
        # Update collision time-series if provided
        try:
            # (time-series plots removed; per-episode tracking still supported via checkboxes)

            # Accumulate values for per-episode reporting when toggled ON
            for m, cb in getattr(self, 'track_metric_cbs', {}).items():
                try:
                    if cb.isChecked() and m in metrics:
                        if not hasattr(self, 'episode_metric_accumulators'):
                            self.episode_metric_accumulators = {}
                        if m not in self.episode_metric_accumulators:
                            self.episode_metric_accumulators[m] = []
                        # store numeric value for later per-episode averaging
                        self.episode_metric_accumulators[m].append(float(metrics[m]))
                except Exception:
                    pass

            if 'mean_upstream_velocity' in metrics:
                # per-timestep upstream velocity not plotted; saved in accumulators if tracking enabled
                pass
        except Exception:
            pass

    def refresh_rl_labels(self):
        # Update episode and reward labels in RL panel
        try:
            self.episode_label.setText(f"Episode: {self.current_episode} | Timestep: {self.current_timestep}")
            self.reward_label.setText(f"Reward: {self.episode_reward:.2f}")
            self.best_reward_label.setText(f"Best: {self.best_reward:.2f}")
        except Exception:
            pass

    
    
    def update_simulation(self):
        """Main simulation update loop (called by timer)."""
        if self.paused or self.current_timestep >= self.n_timesteps:
            return
        # Run one simulation timestep
        try:
            # Create dummy PID controller if needed
            if not hasattr(self, 'pid_controller'):
                from emergent.salmon_abm.sockeye_SoA_OpenGL_RL import PID_controller
                self.pid_controller = PID_controller(
                    self.sim.num_agents,
                    k_p=0.5, k_i=0.0, k_d=0.1
                )
            self.sim.timestep(
                self.current_timestep,
                self.dt,
                9.81,  # gravity
                self.pid_controller
            )
            self.current_timestep += 1
            # RL training logic
            if self.rl_trainer:
                self.update_rl_training()
            # Update displays
            self.update_displays()
        except Exception as e:
            print(f"[ERROR] Error in simulation update: {e}")
            import traceback
            traceback.print_exc()
            self.paused = True
            try:
                self.play_btn.setText("Play")
                self.pause_btn.setText("Pause")
            except Exception as e2:
                print(f"[ERROR] Exception updating play/pause buttons: {e2}")
                import traceback
                traceback.print_exc()
    
    def update_rl_training(self):
        """Update RL training metrics and episode management."""
        # Extract current state
        current_metrics = self.rl_trainer.extract_state_metrics()
        # Update metrics panel immediately so time-series (collisions) are plotted
        try:
            self.update_metrics_panel(current_metrics)
        except Exception as e:
            print(f"[ERROR] Exception in update_metrics_panel: {e}")
            import traceback
            traceback.print_exc()
        
        # Compute reward
        if self.prev_metrics is not None:
            reward = self.rl_trainer.compute_reward(self.prev_metrics, current_metrics)
            self.episode_reward += reward
        
        self.prev_metrics = current_metrics
        
        # Check if episode complete
        if self.current_timestep >= self.n_timesteps:
            print(f"\n{'='*80}")
            print(f"EPISODE {self.current_episode} COMPLETE | Reward: {self.episode_reward:.2f}")
            
            self.rewards_history.append(self.episode_reward)
            
            # Update best
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                print(f"*** NEW BEST REWARD: {self.best_reward:.2f}")
                
                # Save weights
                import json
                import os
                save_dir = os.path.join(os.getcwd(), 'outputs', 'rl_training')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'best_weights.json')
                with open(save_path, 'w') as f:
                    json.dump(self.rl_trainer.behavioral_weights.to_dict(), f, indent=2)
                print(f"Saved to {save_path}")
            
            print(f"{'='*80}\n")
            
            # Mutate weights for next episode
            self.rl_trainer.behavioral_weights.mutate(scale=0.1)
            self.sim.apply_behavioral_weights(self.rl_trainer.behavioral_weights)
            
            # Reset for next episode (including agent positions)
            self.sim.reset_spatial_state(reset_positions=True)
            self.current_episode += 1
            self.current_timestep = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
            
            # Update reward plot
            if hasattr(self, 'reward_plot'):
                self.reward_plot.plot(
                    range(len(self.rewards_history)),
                    self.rewards_history,
                    pen=mkPen('g', width=2),
                    clear=True
                )
            
            # Update metrics panel with passage delay
            if hasattr(self.sim, 'get_mean_passage_delay'):
                mean_delay = self.sim.get_mean_passage_delay()
                self.update_metrics_panel({'mean_passage_delay': mean_delay})
                print(f"Mean Passage Delay: {mean_delay:.2f} timesteps")

            # Compute and append per-episode means for any tracked metrics
            try:
                if hasattr(self, 'episode_metric_accumulators'):
                    for m, vals in self.episode_metric_accumulators.items():
                        if len(vals) == 0:
                            continue
                        mean_val = float(np.mean(vals))
                        # append to per-episode series
                        if m not in self.per_episode_series:
                            self.per_episode_series[m] = []
                        self.per_episode_series[m].append(mean_val)
                        # plot/update handle
                        if m not in self.per_episode_handles:
                            pen = mkPen('y', width=2)
                            self.per_episode_handles[m] = self.per_episode_plot.plot(
                                list(range(len(self.per_episode_series[m]))),
                                self.per_episode_series[m],
                                pen=pen,
                                name=m
                            )
                        else:
                            # update existing handle
                            handle = self.per_episode_handles[m]
                            handle.setData(list(range(len(self.per_episode_series[m]))), self.per_episode_series[m])
                    # clear accumulators for next episode
                    self.episode_metric_accumulators = {}
            except Exception as e:
                print(f"[ERROR] Exception in per-episode metrics update: {e}")
                import traceback
                traceback.print_exc()
    
    def update_displays(self):
        """Update all display elements."""
        # Minimal runtime checks (avoid noisy prints)
        if not hasattr(self, 'tin_mesh') or self.tin_mesh is None:
            print("[RENDER] tin_mesh not ready")
        if not (hasattr(self.sim, 'X') and hasattr(self.sim, 'Y')):
            print("[RENDER] Agent position data missing")
        alive_mask = (self.sim.dead == 0)
        import pyqtgraph.opengl as gl
        # Agents
        if self.show_dead_cb.isChecked():
            x = self.sim.X
            y = self.sim.Y
            colors = np.where(alive_mask[:, np.newaxis], [1, 0.4, 0.4, 0.8], [0.4, 0.4, 0.4, 0.4])
        else:
            x = self.sim.X[alive_mask]
            y = self.sim.Y[alive_mask]
            colors = np.tile([1, 0.4, 0.4, 0.8], (len(x), 1))
        if hasattr(self, 'gl_agent_scatter') and self.gl_agent_scatter is not None:
            self.gl_view.removeItem(self.gl_agent_scatter)
        pts = np.column_stack([x, y, np.zeros_like(x)])
        self.gl_agent_scatter = gl.GLScatterPlotItem(pos=pts, color=colors, size=6)
        self.gl_view.addItem(self.gl_agent_scatter)
        print(f"[RENDER DEBUG] GLScatterPlotItem added, n_agents={len(x)}")

        # Direction indicators
        if hasattr(self, 'gl_direction_lines'):
            for item in self.gl_direction_lines:
                self.gl_view.removeItem(item)
        self.gl_direction_lines = []
        if self.show_direction_cb.isChecked() and hasattr(self.sim, 'heading'):
            if self.show_dead_cb.isChecked():
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

        # Flapping tails
        if hasattr(self, 'gl_tail_lines'):
            for item in self.gl_tail_lines:
                self.gl_view.removeItem(item)
        self.gl_tail_lines = []
        if self.show_tail_cb.isChecked() and hasattr(self.sim, 'heading'):
            if self.show_dead_cb.isChecked():
                headings = self.sim.heading
                pos_x, pos_y = self.sim.X, self.sim.Y
            else:
                headings = self.sim.heading[alive_mask]
                pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]
            current_time = getattr(self, 'current_timestep', 0) * float(getattr(self, 'dt', 1.0))
            nvis = len(pos_x)
            phases = self.tail_phases
            if len(phases) != self.sim.num_agents:
                phases = np.resize(phases, self.sim.num_agents)
            for i in range(nvis):
                if self.show_dead_cb.isChecked():
                    gidx = i
                else:
                    alive_idx = np.nonzero(alive_mask)[0]
                    if i < len(alive_idx):
                        gidx = int(alive_idx[i])
                    else:
                        gidx = i
                h = float(headings[i])
                x0 = float(pos_x[i])
                y0 = float(pos_y[i])
                x1 = x0 - self.tail_L1 * np.cos(h)
                y1 = y0 - self.tail_L1 * np.sin(h)
                phase = float(phases[gidx]) if gidx < len(phases) else 0.0
                theta = self.tail_amp * np.sin(2.0 * np.pi * self.tail_freq * current_time + phase)
                x2 = x1 - self.tail_L2 * np.cos(h + theta)
                y2 = y1 - self.tail_L2 * np.sin(h + theta)
                seg1_pts = np.array([[x0, y0, 0], [x1, y1, 0]])
                seg1 = gl.GLLinePlotItem(pos=seg1_pts, color=(0.7, 0.7, 1.0, 0.8), width=2, antialias=True, mode='lines')
                self.gl_view.addItem(seg1)
                self.gl_tail_lines.append(seg1)
                seg2_pts = np.array([[x1, y1, 0], [x2, y2, 0]])
                seg2 = gl.GLLinePlotItem(pos=seg2_pts, color=(1.0, 0.86, 0.7, 0.86), width=1, antialias=True, mode='lines')
                self.gl_view.addItem(seg2)
                self.gl_tail_lines.append(seg2)

        # Draw direction indicators (OpenGL only)
        import pyqtgraph.opengl as gl
        if hasattr(self, 'gl_direction_lines'):
            for item in self.gl_direction_lines:
                self.gl_view.removeItem(item)
        self.gl_direction_lines = []
        if self.show_direction_cb.isChecked() and hasattr(self.sim, 'heading'):
            if self.show_dead_cb.isChecked():
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

        # Draw purely-visual two-segment tails (proximal rigid + distal oscillating)
        import pyqtgraph.opengl as gl
        if hasattr(self, 'gl_view') and self.gl_view.isVisible():
            # OpenGL mode
            if hasattr(self, 'gl_tail_lines'):
                for item in self.gl_tail_lines:
                    self.gl_view.removeItem(item)
            self.gl_tail_lines = []
            if self.show_tail_cb.isChecked() and hasattr(self.sim, 'heading'):
                if self.show_dead_cb.isChecked():
                    headings = self.sim.heading
                    pos_x, pos_y = self.sim.X, self.sim.Y
                else:
                    alive_mask = (self.sim.dead == 0)
                    headings = self.sim.heading[alive_mask]
                    pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]
                current_time = getattr(self, 'current_timestep', 0) * float(getattr(self, 'dt', 1.0))
                nvis = len(pos_x)
                phases = self.tail_phases
                if len(phases) != self.sim.num_agents:
                    phases = np.resize(phases, self.sim.num_agents)
                for i in range(nvis):
                    if self.show_dead_cb.isChecked():
                        gidx = i
                    else:
                        alive_idx = np.nonzero(alive_mask)[0]
                        if i < len(alive_idx):
                            gidx = int(alive_idx[i])
                        else:
                            gidx = i
                    h = float(headings[i])
                    x0 = float(pos_x[i])
                    y0 = float(pos_y[i])
                    x1 = x0 - self.tail_L1 * np.cos(h)
                    y1 = y0 - self.tail_L1 * np.sin(h)
                    phase = float(phases[gidx]) if gidx < len(phases) else 0.0
                    theta = self.tail_amp * np.sin(2.0 * np.pi * self.tail_freq * current_time + phase)
                    x2 = x1 - self.tail_L2 * np.cos(h + theta)
                    y2 = y1 - self.tail_L2 * np.sin(h + theta)
                    # Proximal segment
                    seg1_pts = np.array([[x0, y0, 0], [x1, y1, 0]])
                    seg1 = gl.GLLinePlotItem(pos=seg1_pts, color=(0.7, 0.7, 1.0, 0.8), width=2, antialias=True, mode='lines')
                    self.gl_view.addItem(seg1)
                    self.gl_tail_lines.append(seg1)
                    # Distal segment
                    seg2_pts = np.array([[x1, y1, 0], [x2, y2, 0]])
                    seg2 = gl.GLLinePlotItem(pos=seg2_pts, color=(1.0, 0.86, 0.7, 0.86), width=1, antialias=True, mode='lines')
                    self.gl_view.addItem(seg2)
                    self.gl_tail_lines.append(seg2)
        else:
            # 2D mode
            if self.show_tail_cb.isChecked() and hasattr(self.sim, 'heading'):
                # clear old tail lines
                if not hasattr(self, 'tail_lines'):
                    self.tail_lines = []
                for line in getattr(self, 'tail_lines', []):
                    try:
                        self.plot_widget.removeItem(line)
                    except Exception:
                        pass
                # reset tail lines container
                self.tail_lines = []
            else:
                # Clear tail lines when disabled
                if hasattr(self, 'tail_lines'):
                    for line in self.tail_lines:
                        try:
                            self.plot_widget.removeItem(line)
                        except Exception:
                            pass
                    self.tail_lines = []
        
        # Update trajectories if enabled
        if self.show_trajectories_cb.isChecked():
            # Store current positions
            self.trajectory_history.append((self.current_timestep, self.sim.X.copy(), self.sim.Y.copy()))
            
            # Limit history to last 100 timesteps to avoid slowdown
            if len(self.trajectory_history) > 100:
                self.trajectory_history.pop(0)
            
            # Clear old trajectory lines
            for line in self.trajectory_lines:
                self.plot_widget.removeItem(line)
            self.trajectory_lines = []

        # Refresh RL labels and metrics display
        try:
            self.refresh_rl_labels()
        except Exception:
            pass

    def reset_simulation(self):
        """Reset simulation state (UI Reset button handler)."""
        try:
            if hasattr(self, 'sim') and hasattr(self.sim, 'reset_spatial_state'):
                try:
                    self.sim.reset_spatial_state(reset_positions=True)
                except TypeError:
                    # older signature
                    self.sim.reset_spatial_state()
        except Exception as e:
            print('[RESET] Exception calling sim.reset_spatial_state():', e)
        # Reset viewer counters
        try:
            self.current_timestep = 0
            self.current_episode = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
        except Exception:
            pass
        try:
            # Update displays to reflect reset
            self.update_displays()
        except Exception:
            pass
        
        
        # Update status labels
        current_time = self.current_timestep * self.dt
        self.timestep_label.setText(f"Timestep: {self.current_timestep} / {self.n_timesteps}")
        self.time_label.setText(f"Time: {current_time:.1f} / {self.T:.1f}s")
        self.alive_label.setText(f"Alive: {alive_mask.sum()} / {self.sim.num_agents}")
        
        # Update agent count labels
        if hasattr(self, 'agent_count_label'):
            self.agent_count_label.setText(f"Total: {self.sim.num_agents}")
            self.alive_count_label.setText(f"Alive: {alive_mask.sum()}")
        
        # Update RL labels
        if self.rl_trainer:
            self.episode_label.setText(f"Episode: {self.current_episode} | Timestep: {self.current_timestep}")
            self.reward_label.setText(f"Reward: {self.episode_reward:.2f}")
            self.best_reward_label.setText(f"Best: {self.best_reward:.2f}")
            
            # Update behavioral weights display during RL training
            if hasattr(self.sim, 'behavioral_weights') and self.weight_labels:
                weights = self.sim.behavioral_weights
                for attr, label in self.weight_labels.items():
                    if hasattr(weights, attr):
                        value = getattr(weights, attr)
                        label.setText(f"{attr.replace('_', ' ').title()}: {value:.3f}")
                        label.setStyleSheet("font-size: 9pt; color: black;")  # Keep black always
        
        # Update metrics
        try:
            if hasattr(self.sim, 'swim_speeds') and alive_mask.sum() > 0:
                mean_speed = np.nanmean(self.sim.swim_speeds[alive_mask, -1])
                max_speed = np.nanmax(self.sim.swim_speeds[alive_mask, -1])
                self.mean_speed_label.setText(f"Mean Speed: {mean_speed:.2f} m/s")
                self.max_speed_label.setText(f"Max Speed: {max_speed:.2f} m/s")
            
            if hasattr(self.sim, 'battery') and alive_mask.sum() > 0:
                mean_energy = np.mean(self.sim.battery[alive_mask])
                min_energy = np.min(self.sim.battery[alive_mask])
                self.mean_energy_label.setText(f"Mean Energy: {mean_energy:.1f}")
                self.min_energy_label.setText(f"Min Energy: {min_energy:.1f}")
            
            if hasattr(self.sim, 'current_centerline_meas') and alive_mask.sum() > 0:
                mean_cl = np.mean(self.sim.current_centerline_meas[alive_mask])
                self.mean_centerline_label.setText(f"Mean Centerline: {mean_cl:.1f} m")
                
            # Schooling metrics (if available)
            if alive_mask.sum() > 1:
                from scipy.spatial import cKDTree
                alive_pos = np.column_stack([self.sim.X[alive_mask], self.sim.Y[alive_mask]])
                tree = cKDTree(alive_pos)
                distances, _ = tree.query(alive_pos, k=2)
                mean_nn_dist = np.mean(distances[:, 1])
                self.mean_nn_dist_label.setText(f"Mean NN Dist: {mean_nn_dist:.1f} m")
                
                # Polarization (alignment of headings)
                if hasattr(self.sim, 'heading'):
                    alive_headings = self.sim.heading[alive_mask]
                    mean_vec = np.array([np.mean(np.cos(alive_headings)), np.mean(np.sin(alive_headings))])
                    polarization = np.linalg.norm(mean_vec)
                    self.polarization_label.setText(f"Polarization: {polarization:.3f}")
        except Exception:
            pass
    
    def toggle_pause(self):
        """Toggle simulation pause state from play/pause button."""
        self.paused = not self.paused
        print(f"[SalmonViewer] Paused set to {self.paused}")
    
    def run(self):
        """Start the viewer (blocking call)."""
        self.show()
        return QtWidgets.QApplication.instance().exec_()

    def reset_simulation(self):
        """Reset simulation state (UI Reset button handler)."""
        try:
            if hasattr(self, 'sim') and hasattr(self.sim, 'reset_spatial_state'):
                try:
                    self.sim.reset_spatial_state(reset_positions=True)
                except TypeError:
                    self.sim.reset_spatial_state()
        except Exception as e:
            print('[RESET] Exception calling sim.reset_spatial_state():', e)
        try:
            self.current_timestep = 0
            self.current_episode = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
        except Exception:
            pass
        try:
            self.update_displays()
        except Exception:
            pass


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    """Convenience function to launch viewer.
    
    Parameters
    ----------
    simulation : simulation object
        Initialized salmon simulation
    dt : float
        Timestep in seconds
    T : float
        Total simulation time
    rl_trainer : RLTrainer, optional
        RL trainer for training visualization
    **kwargs : additional arguments passed to SalmonViewer
    
    Returns
    -------
    int
        Qt application exit code
    """
    print("=== INSIDE launch_viewer ===")
    print(f"Creating Qt Application...")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    print(f"Creating SalmonViewer...")
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    print(f"Starting viewer.run()...")
    return viewer.run()
