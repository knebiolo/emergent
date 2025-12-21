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
try:
    import pyqtgraph as pg
except Exception:
    pg = None


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
        except Exception:
            self.gl_widget = None
        # overlay preview label (hidden by default). This provides a reliable
        # way to display the off-screen FBO contents when the default
        # framebuffer composition fails on some platforms.
        try:
            from PyQt5 import QtGui, QtCore
            self._overlay_label = QtWidgets.QLabel(self)
            self._overlay_label.setVisible(False)
            self._overlay_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self._overlay_label.setStyleSheet('background: rgba(0,0,0,0.5); border: 1px solid #444;')
            self._overlay_label.setAlignment(QtCore.Qt.AlignCenter)
            self._overlay_label.setScaledContents(True)
            self._overlay_label.resize(400, 300)
            # connect renderer signal if the widget exposes it
            try:
                sig = getattr(self.gl_widget, 'fbo_preview_ready', None)
                if sig is not None:
                    sig.connect(self._on_fbo_preview)
            except Exception:
                pass
        except Exception:
            self._overlay_label = None
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
            # agent visualization controls
            self.show_trails_cb = QCheckBox('Show Trails')
            self.show_trails_cb.setChecked(False)
            self.trail_length_label = QLabel('Trail Len: 10')
            self.trail_length_slider = QSlider(Qt.Horizontal)
            self.trail_length_slider.setMinimum(1)
            self.trail_length_slider.setMaximum(200)
            self.trail_length_slider.setValue(10)
            self.agent_size_label = QLabel('Agent Size: 4')
            self.agent_size_slider = QSlider(Qt.Horizontal)
            self.agent_size_slider.setMinimum(1)
            self.agent_size_slider.setMaximum(20)
            self.agent_size_slider.setValue(4)
            # persistence and RL controls
            self.save_previews_cb = QCheckBox('Save Previews')
            self.save_previews_cb.setChecked(False)
            self.auto_mutate_cb = QCheckBox('Auto Mutate Weights')
            self.auto_mutate_cb.setChecked(True)
            # explicit Save Best Weights button
            self.save_best_btn = QPushButton('Save Best Weights')
            self.save_best_btn.clicked.connect(self._on_save_best)
            self.save_weights_btn = QPushButton('Save Weights...')
            self.save_weights_btn.clicked.connect(self._on_save_weights)
            self.load_weights_btn = QPushButton('Load Weights...')
            self.load_weights_btn.clicked.connect(self._on_load_weights)
        except Exception:
            pass
        # RL / episode bookkeeping
        self._current_episode = 0
        self._current_timestep = 0
        self._episode_reward = 0.0
        self._best_reward = float('-inf')
        self.rewards_history = []
        self.episode_metric_accumulators = {}
        self.per_episode_series = {}
        self.per_episode_handles = {}
        # Ensure legacy attributes exist for compatibility/tests even if UI not fully created
        try:
            if not hasattr(self, 'speed_slider'):
                self.speed_slider = None
            if not hasattr(self, 'agent_count_label'):
                self.agent_count_label = QLabel('Total: --')
            if not hasattr(self, 'perim_toggle_btn'):
                self.perim_toggle_btn = None
            if not hasattr(self, 'episode_label'):
                self.episode_label = QLabel('Episode: 0 | Timestep: 0')
            if not hasattr(self, 'reward_plot'):
                self.reward_plot = None
            if not hasattr(self, 'per_episode_plot'):
                self.per_episode_plot = None
            if not hasattr(self, 'last_mesh_payload'):
                self.last_mesh_payload = None
            if not hasattr(self, '_pending_mesh'):
                self._pending_mesh = None
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

    # Compatibility API with original salmon_viewer
    def load_tin_payload(self, payload: dict):
        """Accept a TIN payload dict with keys 'verts','faces','colors' and upload to GL widget."""
        try:
            verts = payload.get('verts')
            faces = payload.get('faces')
            colors = payload.get('colors')
            if verts is None or faces is None or colors is None:
                return False
            if self.gl_widget is not None:
                self.gl_widget.set_mesh(verts, faces, colors)
            else:
                self.last_mesh_payload = {'verts': verts, 'faces': faces, 'colors': colors}
            # store payload for tests/inspection
            try:
                self.last_mesh_payload = payload
            except Exception:
                pass
            return True
        except Exception:
            return False

    def load_tin_payload_dict(self, d: dict):
        """Alias for load_tin_payload to match legacy API used by tests."""
        return self.load_tin_payload(d)

    def load_hecras_mesh(self, hdf_path_or_file, timestep: int = 0, depth_thresh: float | None = 0.05, max_nodes: int | None = 5000, vert_exag: float = 1.0):
        """Extract depth points from a HECRAS HDF5 and upload a mesh to the GL widget.

        This is a convenience method that wraps `hecras_adapter.build_mesh_from_hecras`.
        Returns True on success, False on no-data or failure.
        """
        try:
            from emergent.salmon_abm.viewer_v3.hecras_adapter import build_mesh_from_hecras
            verts, faces, colors = build_mesh_from_hecras(hdf_path_or_file, timestep=timestep, depth_thresh=depth_thresh, max_nodes=max_nodes, vert_exag=vert_exag)
            if verts is None or verts.size == 0:
                return False
            if self.gl_widget is not None:
                try:
                    self.gl_widget.set_mesh(verts, faces, colors)
                except Exception:
                    # GL may not be initialized yet; store for later
                    self._pending_mesh = (verts, faces, colors)
            else:
                # no GL widget available; still store last payload
                self.last_mesh_payload = {'verts': verts, 'faces': faces, 'colors': colors}
            return True
        except Exception:
            return False

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
            # accumulate metrics for RL training if available
            if self.rl_trainer is not None:
                try:
                    # collect state metrics and accumulate selected metrics
                    current_metrics = self.rl_trainer.extract_state_metrics()
                    # update display labels if present
                    try:
                        if 'mean_speed' in current_metrics and hasattr(self, 'mean_speed_label'):
                            self.mean_speed_label.setText(f"Mean Speed: {current_metrics['mean_speed']:.2f}")
                    except Exception:
                        pass
                    # accumulate selected metrics
                    for m, cb in getattr(self, 'track_metric_cbs', {}).items():
                        try:
                            if cb.isChecked():
                                self.episode_metric_accumulators.setdefault(m, []).append(float(current_metrics.get(m, 0.0)))
                        except Exception:
                            pass
                except Exception:
                    pass
            # RL training update if present
            if self.rl_trainer is not None:
                try:
                    self._update_rl_training()
                except Exception:
                    pass
        except Exception:
            pass

    def _update_rl_training(self):
        # Extract current metrics from rl_trainer
        try:
            current_metrics = self.rl_trainer.extract_state_metrics()
        except Exception:
            current_metrics = {}

        # Compute reward increment
        try:
            prev = getattr(self, '_prev_metrics', None)
            if prev is not None:
                reward = self.rl_trainer.compute_reward(prev, current_metrics)
                self._episode_reward = getattr(self, '_episode_reward', 0.0) + float(reward)
            else:
                self._episode_reward = getattr(self, '_episode_reward', 0.0)
        except Exception:
            pass
        self._prev_metrics = current_metrics

        # advance timestep tracking if available
        self._current_timestep = getattr(self.sim, 'current_timestep', getattr(self, '_current_timestep', 0))
        n_timesteps = getattr(self, 'T', getattr(self.sim, 'T', 600))
        if self._current_timestep >= n_timesteps:
            # episode complete
            ep = getattr(self, '_current_episode', 0)
            # save best weights if beat
            try:
                best_reward = getattr(self, '_best_reward', float('-inf'))
                if self._episode_reward > best_reward:
                    self._best_reward = self._episode_reward
                    # save weights
                    import json, os
                    save_dir = getattr(self.sim, 'model_dir', None) or os.getcwd()
                    save_dir = os.path.join(save_dir, 'outputs', 'rl_training')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'best_weights.json')
                    try:
                        json.dump(self.rl_trainer.behavioral_weights.to_dict(), open(save_path, 'w'), indent=2)
                    except Exception:
                        pass
            except Exception:
                pass

            # mutate weights if auto_mutate enabled
            try:
                if getattr(self, 'auto_mutate_cb', None) and self.auto_mutate_cb.isChecked():
                    try:
                        self.rl_trainer.behavioral_weights.mutate(scale=0.1)
                        self.sim.apply_behavioral_weights(self.rl_trainer.behavioral_weights)
                    except Exception:
                        pass
            except Exception:
                pass

            # reset sim spatial state
            try:
                self.sim.reset_spatial_state(reset_positions=True)
            except Exception:
                try:
                    self.sim.reset_spatial_state()
                except Exception:
                    pass

            # advance episode
            self._current_episode = getattr(self, '_current_episode', 0) + 1
            self._current_timestep = 0
            self._episode_reward = 0.0
            self._prev_metrics = None

        # Update UI labels and plots with latest reward / episode info
        try:
            try:
                self.episode_label.setText(f"Episode: {getattr(self, '_current_episode', 0)} | Timestep: {getattr(self, '_current_timestep', 0)}")
            except Exception:
                pass
            try:
                # append reward to history and plot
                self.rewards_history.append(float(getattr(self, '_episode_reward', 0.0)))
                if getattr(self, 'reward_label', None) is not None:
                    self.reward_label.setText(f"Reward: {float(getattr(self, '_episode_reward', 0.0)):.2f}")
                if getattr(self, 'best_reward_label', None) is not None:
                    self.best_reward_label.setText(f"Best: {float(getattr(self, '_best_reward', float('-inf'))):.2f}")
                if getattr(self, 'reward_plot', None) is not None:
                    try:
                        self.reward_plot.clear()
                        self.reward_plot.plot(list(range(len(self.rewards_history))), self.rewards_history, pen=getattr(pg, 'mkPen')( 'g', width=2), clear=True)
                    except Exception:
                        pass
            except Exception:
                pass

            # per-episode metrics plotting
            try:
                for m, series in getattr(self, 'per_episode_series', {}).items():
                    try:
                        # ensure we have a dict of handles
                        if not hasattr(self, 'per_episode_handles'):
                            self.per_episode_handles = {}
                        pen = getattr(pg, 'mkPen')('b', width=1) if pg is not None else None
                        if getattr(self, 'per_episode_plot', None) is not None:
                            # remove previous handle for metric if exists
                            if m in self.per_episode_handles:
                                try:
                                    self.per_episode_plot.removeItem(self.per_episode_handles[m])
                                except Exception:
                                    pass
                            h = None
                            try:
                                h = self.per_episode_plot.plot(list(range(len(series))), series, pen=pen, name=m)
                            except Exception:
                                pass
                            if h is not None:
                                self.per_episode_handles[m] = h
                    except Exception:
                        pass
            except Exception:
                pass
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
        from PyQt5.QtWidgets import QSplitter, QSizePolicy, QHBoxLayout

        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Training & Metrics
        left_panel = QGroupBox('Training & Metrics')
        left_layout = QVBoxLayout()
        self.mean_speed_label = QLabel('Mean Speed: --')
        left_layout.addWidget(self.mean_speed_label)

        # metrics checkboxes
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
            left_layout.addLayout(h)
            self.track_metric_cbs[metric_key] = cb

        try:
            self.collision_count_label = QLabel('Collision Count: --')
            add_label_with_cb(self.collision_count_label, 'collision_count')
            self.mean_upstream_velocity_label = QLabel('Mean Upstream Velocity: --')
            add_label_with_cb(self.mean_upstream_velocity_label, 'mean_upstream_velocity')
            add_label_with_cb(QLabel('Mean Energy: --'), 'energy_efficiency')
            add_label_with_cb(QLabel('Mean Passage Delay: --'), 'mean_passage_delay')
        except Exception:
            pass

        # Agent counters
        try:
            self.agent_count_label = QLabel('Total: --')
            self.alive_count_label = QLabel('Alive: --')
            left_layout.addWidget(self.agent_count_label)
            left_layout.addWidget(self.alive_count_label)
        except Exception:
            pass

        # Reward plots
        try:
            if pg is not None:
                self._reward_plot = pg.PlotWidget(title='Episode Reward')
                self._reward_plot.setMaximumHeight(160)
                self._reward_plot.setLabel('bottom', 'Episode')
                self._reward_plot.setLabel('left', 'Reward')
                self._reward_curve = self._reward_plot.plot([], [], pen=pg.mkPen('g', width=2))
                left_layout.addWidget(self._reward_plot)
            else:
                self._reward_plot = None
                self._reward_curve = None
        except Exception:
            self._reward_plot = None
            self._reward_curve = None

        try:
            if pg is not None:
                self.per_episode_plot = pg.PlotWidget(title='Per-Episode Metrics')
                self.per_episode_plot.setLabel('bottom', 'Episode')
                self.per_episode_plot.setLabel('left', 'Metric Value')
                self.per_episode_plot.setMaximumHeight(220)
                left_layout.addWidget(self.per_episode_plot)
            else:
                self.per_episode_plot = None
        except Exception:
            self.per_episode_plot = None

        left_panel.setLayout(left_layout)

        # Center panel: GL widget or placeholder
        center_container = QtWidgets.QWidget()
        center_layout = QVBoxLayout()
        if self.gl_widget is not None:
            center_layout.addWidget(self.gl_widget)
        # fallback preview label (shows outputs/diag_snapshot_fbo.png)
        try:
            self._fbo_preview_label = QLabel()
            self._fbo_preview_label.setVisible(False)
            center_layout.addWidget(self._fbo_preview_label)
        except Exception:
            self._fbo_preview_label = None
        center_container.setLayout(center_layout)

        # Right panel: Controls & Weights
        right_panel = QGroupBox('Controls & Weights')
        right_layout = QVBoxLayout()
        try:
            right_layout.addWidget(self.play_btn)
            right_layout.addWidget(self.pause_btn)
            right_layout.addWidget(self.reset_btn)
            right_layout.addWidget(self.rebuild_btn)
            # Speed
            try:
                self.speed_label = QLabel('Speed: 1.0x')
                right_layout.addWidget(self.speed_label)
                self.speed_slider = QSlider(Qt.Horizontal)
                self.speed_slider.setMinimum(1)
                self.speed_slider.setMaximum(100)
                self.speed_slider.setValue(10)
                self.speed_slider.valueChanged.connect(self.update_speed)
                right_layout.addWidget(self.speed_slider)
            except Exception:
                pass
            # other controls
            for w in [self.save_best_btn, self.save_weights_btn, self.load_weights_btn, self.ve_label, self.ve_slider,
                      self.show_dead_cb, self.show_direction_cb, self.show_trails_cb, self.trail_length_label,
                      self.trail_length_slider, self.agent_size_label, self.agent_size_slider, self.save_previews_cb,
                      self.auto_mutate_cb]:
                try:
                    right_layout.addWidget(w)
                except Exception:
                    pass
            try:
                self.perim_toggle_btn = QPushButton('Toggle Perimeter')
                self.perim_toggle_btn.clicked.connect(self.toggle_perimeter)
                right_layout.addWidget(self.perim_toggle_btn)
            except Exception:
                pass
        except Exception:
            pass

        # RL status labels
        try:
            self.episode_label = QLabel('Episode: 0 | Timestep: 0')
            self.reward_label = QLabel('Reward: 0.00')
            self.best_reward_label = QLabel('Best: 0.00')
            right_layout.addWidget(self.episode_label)
            right_layout.addWidget(self.reward_label)
            right_layout.addWidget(self.best_reward_label)
            if pg is not None:
                self.reward_plot = pg.PlotWidget(title='Episode Rewards')
                self.reward_plot.setMaximumHeight(160)
                right_layout.addWidget(self.reward_plot)
            else:
                self.reward_plot = None
        except Exception:
            self.reward_plot = None

        right_panel.setLayout(right_layout)

        # helper to show FBO preview into the fallback label
        def show_fbo_preview(visible: bool = True):
            if getattr(self, '_fbo_preview_label', None) is None:
                return
            if visible:
                try:
                    from PyQt5.QtGui import QPixmap
                    import os
                    p = os.path.join(os.getcwd(), 'outputs', 'diag_snapshot_fbo.png')
                    if os.path.exists(p):
                        pix = QPixmap(p)
                        self._fbo_preview_label.setPixmap(pix.scaled(640, 480, Qt.KeepAspectRatio))
                        self._fbo_preview_label.setVisible(True)
                except Exception:
                    pass
            else:
                try:
                    self._fbo_preview_label.setVisible(False)
                except Exception:
                    pass

        # expose helper on instance for tests/quick toggles
        self.show_fbo_preview = show_fbo_preview

        # assemble splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(center_container)
        main_splitter.addWidget(right_panel)
        try:
            main_splitter.setSizes([300, 900, 300])
        except Exception:
            pass

        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        try:
            self.show()
        except Exception:
            pass

        # connect callbacks
        try:
            self.show_trails_cb.stateChanged.connect(self._on_toggle_trails)
            self.trail_length_slider.valueChanged.connect(self._on_trail_length_changed)
            self.agent_size_slider.valueChanged.connect(self._on_agent_size_changed)
            self.show_direction_cb.stateChanged.connect(self._on_toggle_direction)
            try:
                self.speed_slider.valueChanged.connect(self.update_speed)
            except Exception:
                pass
            try:
                self.save_previews_cb.stateChanged.connect(self._on_toggle_save_previews)
            except Exception:
                pass
            try:
                self.save_best_btn.clicked.connect(self._on_save_best)
            except Exception:
                pass
        except Exception:
            pass

        return QtWidgets.QApplication.instance().exec_()

    def _on_toggle_save_previews(self, state: int):
        # placeholder: toggle whether frames are written to disk during episodes
        try:
            self._save_previews = bool(state)
        except Exception:
            self._save_previews = False

    def update_speed(self, v: int):
        try:
            val = v / 10.0
            self.speed_label.setText(f'Speed: {val:.2f}x')
            # if realtime solver present, update its speed multiplier if supported
            if getattr(self, '_rt_solver', None) is not None and hasattr(self._rt_solver, 'set_speed_multiplier'):
                try:
                    self._rt_solver.set_speed_multiplier(val)
                except Exception:
                    pass
        except Exception:
            pass

    def toggle_perimeter(self):
        try:
            if not hasattr(self, 'perim_visible'):
                self.perim_visible = True
            self.perim_visible = not self.perim_visible
            # Actual GL toggle would remove/add perimeter overlay; store state for now
            return self.perim_visible
        except Exception:
            return False

    def rebuild_tin_action(self):
        try:
            QtWidgets.QApplication.instance().processEvents()
            QtWidgets.QTimer.singleShot(10, self.setup_background)
        except Exception:
            try:
                self.setup_background()
            except Exception:
                pass

    def _on_save_best(self):
        try:
            if self.rl_trainer is None:
                return
            import json, os
            save_dir = getattr(self.sim, 'model_dir', None) or os.getcwd()
            save_dir = os.path.join(save_dir, 'outputs', 'rl_training')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'best_weights.json')
            json.dump(self.rl_trainer.behavioral_weights.to_dict(), open(save_path, 'w'), indent=2)
        except Exception:
            pass

    def _on_save_weights(self):
        try:
            if self.rl_trainer is None:
                return
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self, 'Save Weights', 'weights.json', 'JSON Files (*.json)')
            if not path:
                return
            import json
            json.dump(self.rl_trainer.behavioral_weights.to_dict(), open(path, 'w'), indent=2)
        except Exception:
            pass

    def _on_load_weights(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(self, 'Load Weights', '', 'JSON Files (*.json)')
            if not path:
                return
            import json, os
            data = json.load(open(path, 'r'))
            # try to apply via rl_trainer.behavioral_weights.from_dict if available
            if self.rl_trainer is not None and hasattr(self.rl_trainer, 'behavioral_weights'):
                try:
                    self.rl_trainer.behavioral_weights.from_dict(data)
                    if hasattr(self.sim, 'apply_behavioral_weights'):
                        self.sim.apply_behavioral_weights(self.rl_trainer.behavioral_weights)
                    return
                except Exception:
                    pass
            # fallback: if sim supports load_behavioral_weights or similar
            if hasattr(self.sim, 'load_behavioral_weights'):
                try:
                    self.sim.load_behavioral_weights(path)
                    return
                except Exception:
                    pass
        except Exception:
            pass

    def _on_toggle_trails(self, state: int):
        try:
            val = bool(state)
            if self.gl_widget is not None:
                self.gl_widget.set_show_trails(val)
        except Exception:
            pass

    def _on_trail_length_changed(self, v: int):
        try:
            self.trail_length_label.setText(f'Trail Len: {v}')
            if self.gl_widget is not None:
                self.gl_widget.set_trail_length(int(v))
        except Exception:
            pass

    def _on_agent_size_changed(self, v: int):
        try:
            self.agent_size_label.setText(f'Agent Size: {v}')
            if self.gl_widget is not None:
                self.gl_widget.set_point_size(float(v))
        except Exception:
            pass

    def _on_toggle_direction(self, state: int):
        try:
            val = bool(state)
            if self.gl_widget is not None:
                self.gl_widget.set_show_directions(val)
        except Exception:
            pass


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
