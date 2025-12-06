"""
Cleaned SalmonViewer implementation (rebuilt).

This preserves the original UI panels, GL-based TIN rendering, HECRAS perimeter
inference hook, RL panels, and weight controls. It is intentionally conservative
"""
Minimal SalmonViewer replacement (clean start).

Provides:
- _GLMeshBuilder: QThread that emits mesh payloads
- SalmonViewer: compact QWidget with GL view (if available) and basic controls
- launch_viewer: convenience to create QApplication and start the viewer

This intentionally omits advanced metric panels to ensure a clean baseline.
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QSlider, QGroupBox, QCheckBox

import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None


class _GLMeshBuilder(QtCore.QThread):
    mesh_ready = QtCore.pyqtSignal(object)

    def __init__(self, tri_pts, tri_vals, tris, vert_exag=1.0, parent=None):
        super().__init__(parent=parent)
        self.tri_pts = np.asarray(tri_pts, dtype=float)
        self.tri_vals = np.asarray(tri_vals, dtype=float)
        self.tris = np.asarray(tris, dtype=np.int32)
        self.vert_exag = float(vert_exag)

    def run(self):
        try:
            pts = self.tri_pts
            vals = self.tri_vals
            faces = self.tris
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

            payload = {"verts": verts.astype(float), "faces": faces.astype(int), "colors": colors.astype(float)}
            self.mesh_ready.emit(payload)
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
        self.setWindowTitle('SalmonViewer')
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

            from scipy.spatial import Delaunay
            tri = Delaunay(pts)
            tris = tri.simplices

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

            builder = _GLMeshBuilder(pts, vals, tris, vert_exag=getattr(self.sim, 'vert_exag', 1.0), parent=self)

            def _on_mesh(payload):
                if 'error' in payload:
                    print('[TIN] builder error', payload['error'])
                    return
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
            self.sim.timestep(self.current_timestep, self.dt, 9.81, None)
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

                                                builder = _GLMeshBuilder(pts, vals, tris, vert_exag=getattr(self.sim, 'vert_exag', 1.0), parent=self)

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

                                            print('[RENDER] No HECRAS data; skipping background TIN build')
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
                                            self.sim.timestep(self.current_timestep, self.dt, 9.81, None)
                                            self.current_timestep += 1
                                            self.update_displays()
                                        except Exception as e:
                                            print('[ERROR] update_simulation failed:', e)
                                            import traceback
                                            traceback.print_exc()
                                            self.paused = True

                                    def update_displays(self):
                                        try:
                                            if gl is None or self.gl_view is None:
                                                return
                                            if not (hasattr(self.sim, 'X') and hasattr(self.sim, 'Y')):
                                                return
                                            alive_mask = (getattr(self.sim, 'dead', np.zeros(getattr(self.sim, 'num_agents', 0))) == 0)
                                            if getattr(self.sim, 'num_agents', 0) == 0:
                                                return
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
                                    print('=== INSIDE launch_viewer ===')
                                    print('Creating Qt Application...')
                                    app = QtWidgets.QApplication.instance()
                                    if app is None:
                                        app = QtWidgets.QApplication(sys.argv)
                                    print('Creating SalmonViewer...')
                                    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
                                    print('Starting viewer.run()...')
                                    return viewer.run()


                                # End of minimal viewer
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
