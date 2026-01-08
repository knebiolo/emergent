#!/usr/bin/env python3
"""
Real-time RL training visualizer for behavioral weight optimization.

Three-panel Qt interface:
- Left panel: Behavioral weights and diagnostics
- Center panel: Simulation playback (similar to realtime_viewer)
- Right panel: Training controls (play/pause/stop, generation settings, etc.)

Usage:
    python -m emergent.salmon_abm.rl_training_viewer --model-dir data/salmon_abm --start-polygon data/salmon_abm/start_loc_river_right.shp
    
    # Or programmatically
    from emergent.salmon_abm.rl_training_viewer import RLTrainingViewer
    viewer = RLTrainingViewer()
    viewer.start_training(...)
"""
from __future__ import annotations

import sys
import os
import time
import threading
import argparse
import multiprocessing as mp
import queue as queue_mod
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import numpy as np

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QPushButton,
        QSlider,
        QLabel,
        QHBoxLayout,
        QVBoxLayout,
        QGridLayout,
        QGroupBox,
        QSpinBox,
        QDoubleSpinBox,
        QTextEdit,
        QSplitter,
        QOpenGLWidget,
        QProgressBar,
        QLineEdit,
        QFileDialog,
        QCheckBox,
        QComboBox,
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QPainter, QColor, QPen, QFont
except ImportError as e:
    raise ImportError(f"PyQt5 is required for RL training visualizer: {e}")

# Import RL training components
try:
    from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer
    from emergent.salmon_abm.simulation import simulation
    from emergent.salmon_abm.realtime_viewer import ReplayWidget
except ImportError as e:
    raise ImportError(f"Could not import RL training components: {e}")


def _drain_control_messages(control_queue, paused: bool, stopped: bool) -> Tuple[bool, bool]:
    """Return (paused, stopped) after draining control messages."""
    while True:
        try:
            msg = control_queue.get_nowait()
        except queue_mod.Empty:
            break
        if msg == "stop":
            stopped = True
            paused = False
        elif msg == "pause":
            paused = True
        elif msg == "resume":
            paused = False
    return paused, stopped


def _training_process_main(config: Dict[str, Any], output_queue, control_queue) -> None:
    """Run RL training in a subprocess to avoid UI/GIL contention."""
    import traceback
    import numpy as np
    from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer, compute_episode_reward
    from emergent.salmon_abm.simulation import simulation

    try:
        env_files = config.get("env_files") or []
        if not env_files:
            raise ValueError("No environment files provided for training process")

        longitudinal_path = config.get("longitudinal_path")
        if not longitudinal_path or not os.path.exists(longitudinal_path):
            raise FileNotFoundError(f"Longitudinal profile shapefile required but not found: {longitudinal_path}")

        num_episodes = int(config.get("num_episodes", 1))
        num_timesteps = int(config.get("num_timesteps", 50))
        num_agents = int(config.get("num_agents", 200))
        exploration_noise = float(config.get("exploration_noise", 0.1))
        reward_weights = config.get("reward_weights") or None
        body_length = float(config.get("body_length", 0.3))
        dt = float(config.get("dt", 1.0))

        initial_weights = BehavioralWeights.from_dict(config.get("initial_weights") or {})

        def simulation_factory(weights: BehavioralWeights):
            sim = simulation(
                model_dir=config.get("model_dir"),
                model_name=config.get("model_name"),
                crs=config.get("crs"),
                basin=config.get("basin"),
                water_temp=float(config.get("water_temp", 12.0)),
                start_polygon=config.get("start_polygon"),
                env_files=env_files,
                longitudinal_profile=longitudinal_path,
                fish_length=None,
                num_timesteps=num_timesteps,
                num_agents=num_agents,
                use_gpu=False,
                pid_tuning=False,
                db_path=None,
                output_write_mode="none",
                output_write_backend="sync",
            )

            sim.load_behavioral_weights(weights_dict=weights.to_dict())

            fish_length_m = 0.3
            sensory_range = 5.0
            sim.neighbor_buffer_radius = sensory_range * fish_length_m
            sim.neighbor_buffer_lengths = sensory_range

            sim.heading = np.random.uniform(0, 2 * np.pi, sim.num_agents).astype(np.float32)
            sim.sog = np.random.uniform(0.1, 1.5, sim.num_agents).astype(np.float32)
            return sim

        trainer = RLTrainer(
            simulation_factory=simulation_factory,
            initial_weights=initial_weights,
            config={
                "exploration_noise": exploration_noise,
                "body_length": body_length,
                "dt": dt,
                "num_timesteps": num_timesteps,
                "reward_weights": reward_weights,
            },
        )

        longitudinal_profile = None
        try:
            sim = trainer.simulation_factory(trainer.initial_weights)
            longitudinal_profile = getattr(sim, "longitudinal", None)
            sim.close()
        except Exception as e:
            raise RuntimeError(f"Failed to cache longitudinal profile: {e}") from e

        if longitudinal_profile is None:
            raise ValueError("Longitudinal profile not loaded from simulation")

        current_weights = trainer.initial_weights
        paused = False
        stopped = False

        for episode in range(num_episodes):
            paused, stopped = _drain_control_messages(control_queue, paused, stopped)
            if stopped:
                break
            while paused and not stopped:
                time.sleep(0.1)
                paused, stopped = _drain_control_messages(control_queue, paused, stopped)
            if stopped:
                break

            output_queue.put({"type": "episode_started", "episode": episode, "total": num_episodes})

            positions, headings, velocities, battery, alive, velocity_field = trainer.run_episode(current_weights)

            reward, components = compute_episode_reward(
                positions,
                headings,
                velocities,
                alive,
                body_length=trainer.body_length,
                threat_level=current_weights.threat_level,
                behavioral_weights=current_weights.to_dict(),
                battery_history=battery,
                longitudinal_profile=longitudinal_profile,
                velocity_field_history=velocity_field,
                reward_weights=trainer.reward_weights,
            )

            if reward > trainer.best_reward:
                trainer.best_reward = reward
                trainer.best_weights = current_weights

            trainer.episode_history.append((episode, float(reward)))

            output_queue.put(
                {
                    "type": "episode_computed",
                    "episode": episode,
                    "reward": float(reward),
                    "components": components,
                    "positions": positions,
                    "headings": headings,
                    "battery": battery,
                    "alive": alive,
                    "weights": current_weights.to_dict(),
                    "best_reward": float(trainer.best_reward),
                }
            )

            current_weights = trainer.best_weights.mutate(mutation_scale=trainer.exploration_noise)

        if stopped:
            output_queue.put(
                {
                    "type": "training_stopped",
                    "best_weights": trainer.best_weights.to_dict(),
                    "history": trainer.episode_history,
                }
            )
        else:
            output_queue.put(
                {
                    "type": "training_completed",
                    "best_weights": trainer.best_weights.to_dict(),
                    "history": trainer.episode_history,
                }
            )
    except Exception as exc:
        msg = f"Training error: {exc}\n{traceback.format_exc()}"
        try:
            output_queue.put({"type": "error", "message": msg})
        finally:
            raise

class SimulationCanvas(QWidget):
    """
    Center panel: Real-time simulation visualization using ReplayWidget.
    
    Wrapper around ReplayWidget that handles live position updates during training.
    """
    
    # Signal emitted when animation completes
    animation_finished = pyqtSignal()
    
    def __init__(self, parent=None, model_dir: Optional[str] = None):
        super().__init__(parent)
        
        # Initialize with dummy positions (1 timestep, 1 agent)
        dummy_positions = np.zeros((1, 1, 2), dtype=np.float32)
        
        # Try to load depth and velocity rasters for background
        env_depth = None
        self.vel_x_data = None
        self.vel_y_data = None
        self.vel_transform = None
        self.vel_bbox = None
        
        if model_dir and os.path.exists(model_dir):
            depth_path = os.path.join(model_dir, 'depth.tif')
            if os.path.exists(depth_path):
                env_depth = depth_path
            
            # Load velocity field for arrow rendering
            vel_x_path = os.path.join(model_dir, 'vel_x.tif')
            vel_y_path = os.path.join(model_dir, 'vel_y.tif')
            if os.path.exists(vel_x_path) and os.path.exists(vel_y_path):
                try:
                    from emergent.salmon_abm import io as _io
                    vel_x_arr, vel_x_transform, _ = _io.enviro_import(vel_x_path)
                    vel_y_arr, vel_y_transform, _ = _io.enviro_import(vel_y_path)
                    
                    self.vel_x_data = np.array(vel_x_arr, dtype=float)
                    self.vel_y_data = np.array(vel_y_arr, dtype=float)
                    self.vel_transform = vel_x_transform
                    
                    # Compute velocity raster bbox
                    h, w = self.vel_x_data.shape
                    try:
                        a_t, b_t, c_t, d_t, e_t, f_t = vel_x_transform
                    except:
                        t = vel_x_transform
                        a_t, b_t, c_t, d_t, e_t, f_t = (t.a, t.b, t.c, t.d, t.e, t.f)
                    
                    xs = [a_t * c + b_t * r + c_t for c in [0, w] for r in [0, h]]
                    ys = [d_t * c + e_t * r + f_t for c in [0, w] for r in [0, h]]
                    self.vel_bbox = (min(xs), max(xs), min(ys), max(ys))
                except Exception as e:
                    print(f"Warning: Could not load velocity fields: {e}")
        
        # Create ReplayWidget with environment background
        self.replay_widget = ReplayWidget(
            positions=dummy_positions,
            parent=self,
            env_depth=env_depth,
            pad=1.15,
            point_size=4.0
        )
        
        # Pass velocity data to replay widget for arrow rendering
        if self.vel_x_data is not None:
            self.replay_widget._vel_x_data = self.vel_x_data
            self.replay_widget._vel_y_data = self.vel_y_data
            self.replay_widget._vel_bbox = self.vel_bbox
            self.replay_widget._vel_transform = self.vel_transform
        
        # Connect to replay widget's timer to detect when animation finishes
        self._animation_finished_emitted = False
        self._animation_watchdog = QTimer(self)
        self._animation_watchdog.setInterval(100)
        self._animation_watchdog.timeout.connect(self._poll_animation_complete)
        self.replay_widget.timer.timeout.connect(self._check_animation_complete)
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.replay_widget)
        self.setLayout(layout)
        
    def _check_animation_complete(self):
        """Check if animation reached the end and emit signal."""
        if self.replay_widget.frame >= self.replay_widget.T - 1:
            # Animation reached the end
            self.replay_widget.playing = False
            if self.replay_widget.timer.isActive():
                self.replay_widget.timer.stop()
            self._emit_animation_finished()

    def _poll_animation_complete(self):
        """Fallback watchdog in case timer callbacks miss the last frame."""
        if self._animation_finished_emitted:
            return
        if self.replay_widget.frame >= self.replay_widget.T - 1 and not self.replay_widget.timer.isActive():
            self._emit_animation_finished()

    def _emit_animation_finished(self):
        """Emit animation_finished once per episode."""
        if self._animation_finished_emitted:
            return
        self._animation_finished_emitted = True
        if self._animation_watchdog.isActive():
            self._animation_watchdog.stop()
        self.animation_finished.emit()
    
    def wheelEvent(self, event):
        """Forward wheel events to replay widget for zooming only."""
        # Forward to replay_widget and prevent default scroll behavior
        self.replay_widget.wheelEvent(event)
        event.accept()
        
    def set_positions(self, positions: np.ndarray, headings: Optional[np.ndarray] = None, battery: Optional[np.ndarray] = None, alive: Optional[np.ndarray] = None):
        """
        Update agent positions for visualization.
        
        Args:
            positions: Array of shape (num_agents, 2) or (num_timesteps, num_agents, 2)
            headings: Optional array of headings for oriented rendering
            battery: Optional array of battery levels for color visualization (green=full, red=depleted)
            alive: Optional boolean array indicating which agents are alive (for dead fish coloring)
        """
        print(f"[RL VIEWER DEBUG] set_positions called: positions shape={positions.shape if positions is not None else None}", flush=True)
        if positions is None or positions.size == 0:
            print("[RL VIEWER DEBUG] Positions empty, returning early", flush=True)
            return
            
        # Ensure positions are 3D (T, N, 2)
        if positions.ndim == 2:
            # Single timestep: (N, 2) -> (1, N, 2)
            positions = positions[np.newaxis, :, :]
        
        # Update ReplayWidget with full trajectory
        self.replay_widget.positions = positions
        self.replay_widget.T, self.replay_widget.N, _ = positions.shape
        
        # Update headings if provided
        if headings is not None:
            if headings.ndim == 1:
                headings = headings[np.newaxis, :]
            self.replay_widget.heading_array = headings
        
        # Update battery if provided (for color visualization)
        if battery is not None:
            if battery.ndim == 1:
                battery = battery[np.newaxis, :]
            self.replay_widget.battery_array = battery
        
        # Update alive status if provided (for dead fish coloring)
        if alive is not None:
            if alive.ndim == 1:
                alive = alive[np.newaxis, :]
            self.replay_widget.alive_array = alive
        
        # Update bounds to include all positions in trajectory
        xs = positions[:, :, 0]
        ys = positions[:, :, 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.any(valid):
            self.replay_widget.xmin = float(np.nanmin(xs[valid]))
            self.replay_widget.xmax = float(np.nanmax(xs[valid]))
            self.replay_widget.ymin = float(np.nanmin(ys[valid]))
            self.replay_widget.ymax = float(np.nanmax(ys[valid]))
        
        # Start from first frame and auto-play the trajectory
        print(f"[RL VIEWER DEBUG] Starting animation: T={self.replay_widget.T}, N={self.replay_widget.N}, frame=0", flush=True)
        self.replay_widget.frame = 0
        self.replay_widget.playing = True
        self.replay_widget.start()  # Start animation
        self.replay_widget.update()
        self._animation_finished_emitted = False
        if not self._animation_watchdog.isActive():
            self._animation_watchdog.start()
        print(f"[RL VIEWER DEBUG] Animation started, playing={self.replay_widget.playing}", flush=True)


class WeightsPanel(QWidget):
    """
    Left panel: Display behavioral weights and training diagnostics.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Behavioral Weights")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Weights display (editable spin boxes)
        self.weights_group = QGroupBox("Current Weights (Editable)")
        self.weights_layout = QVBoxLayout()
        self.weight_spinboxes = {}  # Changed from weight_labels to weight_spinboxes
        self.weights_group.setLayout(self.weights_layout)
        layout.addWidget(self.weights_group)
        
        # Arbitration Order display
        self.order_group = QGroupBox("Cue Application Order")
        self.order_layout = QVBoxLayout()
        self.order_labels = {}
        self.order_group.setLayout(self.order_layout)
        layout.addWidget(self.order_group)
        
        # Diagnostics
        self.diagnostics_group = QGroupBox("Training Diagnostics")
        diag_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0 / 0")
        self.reward_label = QLabel("Current Reward: 0.00")
        self.best_reward_label = QLabel("Best Reward: 0.00")
        self.improvement_label = QLabel("Improvement: 0.00")
        
        diag_layout.addWidget(self.episode_label)
        diag_layout.addWidget(self.reward_label)
        diag_layout.addWidget(self.best_reward_label)
        diag_layout.addWidget(self.improvement_label)
        
        self.diagnostics_group.setLayout(diag_layout)
        layout.addWidget(self.diagnostics_group)
        
        # Reward components
        self.components_group = QGroupBox("Reward Components")
        self.components_layout = QVBoxLayout()
        self.component_labels = {}
        self.components_group.setLayout(self.components_layout)
        layout.addWidget(self.components_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_weights(self, weights: BehavioralWeights):
        """Update displayed weights with editable spin boxes."""
        print(f"[UPDATE_WEIGHTS DEBUG] Called with weights, shallow_weight={weights.shallow_weight}, cohesion_weight={weights.cohesion_weight}", flush=True)
        
        # Clear ALL existing widgets and layouts properly
        # First, delete all child widgets and layouts
        while self.weights_layout.count():
            item = self.weights_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Clear nested layout
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                item.layout().deleteLater()
        
        self.weight_spinboxes.clear()
        
        # Add new weight spin boxes
        weights_dict = weights.to_dict()
        print(f"[UPDATE_WEIGHTS DEBUG] weights_dict has {len(weights_dict)} items", flush=True)
        
        # Weight attributes to display (skip order fields and non-weight params)
        weight_keys = [
            'shallow_weight', 'border_cue_weight', 'avoid_weight', 'collision_weight',
            'alignment_weight', 'cohesion_weight', 'low_speed_weight', 'refugia_weight',
            'rheotaxis_weight', 'wave_drag_weight', 'arbitration_tolerance'
        ]
        
        for name in weight_keys:
            if name not in weights_dict:
                continue
                
            value = weights_dict[name]
            
            # Create horizontal layout for label + spinbox
            row_layout = QHBoxLayout()
            
            # Label
            label_text = name.replace('_weight', '').replace('_', ' ').title()
            label = QLabel(f"{label_text}:")
            label.setMinimumWidth(150)
            row_layout.addWidget(label)
            
            # Spin box
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 1000000)
            spinbox.setDecimals(1)
            spinbox.setValue(float(value))
            spinbox.setSingleStep(100.0 if name == 'arbitration_tolerance' else 1000.0)
            spinbox.setObjectName(name)  # Store attribute name
            
            # Connect to update function
            spinbox.valueChanged.connect(lambda v, attr=name: self._on_weight_changed(attr, v))
            
            row_layout.addWidget(spinbox)
            self.weights_layout.addLayout(row_layout)
            self.weight_spinboxes[name] = spinbox
            print(f"[UPDATE_WEIGHTS DEBUG] Added spinbox: {name}={value}", flush=True)
    
    def _on_weight_changed(self, attr_name: str, value: float):
        """Called when a weight spinbox value changes."""
        print(f"[WEIGHT CHANGED] {attr_name} = {value}", flush=True)
        # The parent window will handle applying these changes to current_weights
    
    def get_edited_weights(self, base_weights: BehavioralWeights) -> BehavioralWeights:
        """Get a new BehavioralWeights object with values from the spin boxes."""
        weights_dict = base_weights.to_dict()
        
        # Update with edited values from spin boxes
        for name, spinbox in self.weight_spinboxes.items():
            weights_dict[name] = spinbox.value()
        
        # Create new weights object from updated dict
        from emergent.salmon_abm.rl_training import BehavioralWeights
        return BehavioralWeights.from_dict(weights_dict)
    
    def update_diagnostics(self, episode: int, total_episodes: int, reward: float, 
                          best_reward: float, initial_reward: float):
        """Update training diagnostics."""
        self.episode_label.setText(f"Episode: {episode} / {total_episodes}")
        self.reward_label.setText(f"Current Reward: {reward:.2f}")
        self.best_reward_label.setText(f"Best Reward: {best_reward:.2f}")
        improvement = reward - initial_reward
        self.improvement_label.setText(f"Improvement: {improvement:+.2f}")
    
    def clear_diagnostics(self):
        """Clear all diagnostic displays."""
        self.episode_label.setText("Episode: 0 / 0")
        self.reward_label.setText("Current Reward: 0.00")
        self.best_reward_label.setText("Best Reward: 0.00")
        self.improvement_label.setText("Improvement: +0.00")
        
        # Clear component labels
        for label in self.component_labels.values():
            self.components_layout.removeWidget(label)
            label.deleteLater()
        self.component_labels.clear()
        
    def update_components(self, components: Dict[str, float]):
        """Update reward component breakdown."""
        # Clear existing labels
        for label in self.component_labels.values():
            self.components_layout.removeWidget(label)
            label.deleteLater()
        self.component_labels.clear()
        
        # Add component labels
        for name, value in sorted(components.items()):
            label = QLabel(f"{name}: {value:.2f}")
            label.setFont(QFont("Courier New", 9))
            self.components_layout.addWidget(label)
            self.component_labels[name] = label
    
    def update_order(self, order_dict: Dict[int, str], weights: Optional['BehavioralWeights'] = None):
        """Update displayed arbitration order with optional remapping.
        
        Args:
            order_dict: Default mapping of position -> cue_name
            weights: Optional BehavioralWeights with order_0-9 fields showing new positions
        """
        # Clear existing labels
        for label in self.order_labels.values():
            self.order_layout.removeWidget(label)
            label.deleteLater()
        self.order_labels.clear()
        
        # Get remapping if weights provided
        if weights is not None:
            # Extract order remapping from weights
            new_positions = [
                weights.order_0, weights.order_1, weights.order_2, weights.order_3, weights.order_4,
                weights.order_5, weights.order_6, weights.order_7, weights.order_8, weights.order_9
            ]
        else:
            new_positions = None
        
        # Add order labels
        for position, cue_name in sorted(order_dict.items()):
            if new_positions is not None:
                new_pos = new_positions[position]
                if new_pos != position:
                    label_text = f"{position}: {cue_name} → {new_pos}"
                else:
                    label_text = f"{position}: {cue_name}"
            else:
                label_text = f"{position}: {cue_name}"
            
            label = QLabel(label_text)
            label.setFont(QFont("Courier New", 9))
            self.order_layout.addWidget(label)
            self.order_labels[position] = label


class ControlPanel(QWidget):
    """
    Right panel: Training controls and settings.
    """
    
    # Signals
    start_training = pyqtSignal()
    pause_training = pyqtSignal()
    stop_training = pyqtSignal()
    randomize_weights = pyqtSignal()
    set_blanket_value = pyqtSignal()  # NEW: Set all weights to uniform value
    randomize_order = pyqtSignal()
    reset_training = pyqtSignal()  # NEW: Reset to initial state
    episode_selected = pyqtSignal(int)  # NEW: User selected episode to replay
    next_episode = pyqtSignal()  # NEW: Show next episode
    prev_episode = pyqtSignal()  # NEW: Show previous episode
    toggle_playback = pyqtSignal()  # Play/pause animation
    restart_animation = pyqtSignal()  # Restart current episode animation
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Controls")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Training control buttons
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("▶ Start Training")
        self.btn_pause = QPushButton("⏸ Pause Training")
        self.btn_stop = QPushButton("⏹ Stop Training")
        
        self.btn_start.clicked.connect(self.start_training.emit)
        self.btn_pause.clicked.connect(self.pause_training.emit)
        self.btn_stop.clicked.connect(self.stop_training.emit)
        
        train_layout.addWidget(self.btn_start)
        train_layout.addWidget(self.btn_pause)
        train_layout.addWidget(self.btn_stop)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Animation playback controls
        playback_group = QGroupBox("Animation Playback")
        playback_layout = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("⏸ Pause")
        self.btn_play_pause.setToolTip("Play/Pause current episode animation")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        
        self.btn_restart_anim = QPushButton("↺ Restart")
        self.btn_restart_anim.setToolTip("Restart current episode from beginning")
        self.btn_restart_anim.clicked.connect(self.restart_animation)
        
        playback_layout.addWidget(self.btn_play_pause)
        playback_layout.addWidget(self.btn_restart_anim)
        playback_group.setLayout(playback_layout)
        layout.addWidget(playback_group)
        
        # Episode navigation
        nav_group = QGroupBox("Episode Navigation")
        nav_layout = QVBoxLayout()
        
        # Dropdown + buttons row
        nav_controls = QHBoxLayout()
        
        self.btn_prev_episode = QPushButton("◀")
        self.btn_prev_episode.setMaximumWidth(40)
        self.btn_prev_episode.setToolTip("Show previous episode")
        self.btn_prev_episode.clicked.connect(self.prev_episode.emit)
        nav_controls.addWidget(self.btn_prev_episode)
        
        self.episode_combo = QComboBox()
        self.episode_combo.setToolTip("Select episode to replay")
        self.episode_combo.currentIndexChanged.connect(self._on_episode_combo_changed)
        nav_controls.addWidget(self.episode_combo)
        
        self.btn_next_episode = QPushButton("▶")
        self.btn_next_episode.setMaximumWidth(40)
        self.btn_next_episode.setToolTip("Show next episode")
        self.btn_next_episode.clicked.connect(self.next_episode.emit)
        nav_controls.addWidget(self.btn_next_episode)
        
        nav_layout.addLayout(nav_controls)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        # Episodes
        params_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 100000)
        self.episodes_spin.setValue(50)
        self.episodes_spin.setToolTip("Number of training episodes to run. Each episode tests one set of behavioral weights.")
        params_layout.addWidget(self.episodes_spin, 0, 1)
        
        # Timesteps per episode
        params_layout.addWidget(QLabel("Timesteps:"), 1, 0)
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(10, 100000)
        self.timesteps_spin.setValue(100)
        self.timesteps_spin.setToolTip("Number of simulation timesteps per episode. Large values increase compute and memory.")
        params_layout.addWidget(self.timesteps_spin, 1, 1)
        
        # Number of agents
        params_layout.addWidget(QLabel("Agents:"), 2, 0)
        self.agents_spin = QSpinBox()
        self.agents_spin.setRange(10, 50000)
        self.agents_spin.setValue(200)
        self.agents_spin.setToolTip("Number of fish agents in the simulation. Large values can be slow and memory-heavy.")
        params_layout.addWidget(self.agents_spin, 2, 1)
        
        # Exploration noise
        params_layout.addWidget(QLabel("Exploration:"), 3, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 1.0)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.1)
        self.noise_spin.setToolTip("Mutation scale for exploring new behavioral weights (0.1 = 10% random variation). Higher values = more exploration, lower = more exploitation of good weights.")
        params_layout.addWidget(self.noise_spin, 3, 1)
        
        # Storage interval (memory optimization)
        params_layout.addWidget(QLabel("Store every Nth:"), 4, 0)
        self.storage_interval_spin = QSpinBox()
        self.storage_interval_spin.setRange(1, 10000)
        self.storage_interval_spin.setValue(10)
        self.storage_interval_spin.setToolTip("Store trajectory data for replay every N episodes (plus first 5 and best). 1=all episodes, 10=every 10th. Saves memory for long training runs.")
        params_layout.addWidget(self.storage_interval_spin, 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Blanket value controls (above randomize for better flow)
        blanket_layout = QHBoxLayout()
        self.blanket_value_input = QLineEdit("10.0")
        self.blanket_value_input.setMaximumWidth(60)
        self.blanket_value_input.setToolTip("Value to set for all behavioral weights")
        blanket_layout.addWidget(QLabel("Blanket value:"))
        blanket_layout.addWidget(self.blanket_value_input)
        
        self.btn_set_blanket = QPushButton("📋 Set All Weights")
        self.btn_set_blanket.clicked.connect(self.set_blanket_value.emit)
        self.btn_set_blanket.setToolTip("Set all behavioral weights to the specified blanket value. Only works before training starts.")
        blanket_layout.addWidget(self.btn_set_blanket)
        blanket_layout.addStretch()
        
        layout.addLayout(blanket_layout)
        
        # Reward weights section (SEPARATE from behavioral weights - controls scoring, not training)
        reward_group = QGroupBox("Reward Weights (Objective Function)")
        reward_layout = QGridLayout()
        
        # Three columns: Label, Spinbox | Label, Spinbox | Label, Spinbox
        # Column 1: Positive rewards
        reward_layout.addWidget(QLabel("<b>Positive:</b>"), 0, 0, 1, 2)
        
        reward_layout.addWidget(QLabel("Upstream:"), 1, 0)
        self.upstream_weight_spin = QDoubleSpinBox()
        self.upstream_weight_spin.setRange(0, 100)
        self.upstream_weight_spin.setValue(10.0)
        self.upstream_weight_spin.setDecimals(2)
        self.upstream_weight_spin.setToolTip("Multiplier for meters traveled upstream (PRIMARY)")
        reward_layout.addWidget(self.upstream_weight_spin, 1, 1)
        
        reward_layout.addWidget(QLabel("Cohesion:"), 2, 0)
        self.cohesion_reward_spin = QDoubleSpinBox()
        self.cohesion_reward_spin.setRange(0, 100)
        self.cohesion_reward_spin.setValue(0.001)
        self.cohesion_reward_spin.setDecimals(4)
        self.cohesion_reward_spin.setToolTip("Multiplier for cohesion score sum")
        reward_layout.addWidget(self.cohesion_reward_spin, 2, 1)
        
        reward_layout.addWidget(QLabel("Alignment:"), 3, 0)
        self.alignment_reward_spin = QDoubleSpinBox()
        self.alignment_reward_spin.setRange(0, 100)
        self.alignment_reward_spin.setValue(0.01)
        self.alignment_reward_spin.setDecimals(4)
        self.alignment_reward_spin.setToolTip("Multiplier for alignment score sum")
        reward_layout.addWidget(self.alignment_reward_spin, 3, 1)
        
        reward_layout.addWidget(QLabel("Energy eff:"), 4, 0)
        self.energy_reward_spin = QDoubleSpinBox()
        self.energy_reward_spin.setRange(0, 100)
        self.energy_reward_spin.setValue(2.0)
        self.energy_reward_spin.setDecimals(2)
        self.energy_reward_spin.setToolTip("Multiplier for distance/speed² ratio")
        reward_layout.addWidget(self.energy_reward_spin, 4, 1)
        
        reward_layout.addWidget(QLabel("Drafting:"), 5, 0)
        self.drafting_reward_spin = QDoubleSpinBox()
        self.drafting_reward_spin.setRange(0, 100)
        self.drafting_reward_spin.setValue(20.0)
        self.drafting_reward_spin.setDecimals(1)
        self.drafting_reward_spin.setToolTip("Multiplier for formation benefits (disabled)")
        reward_layout.addWidget(self.drafting_reward_spin, 5, 1)
        
        # Column 2: Penalties (negative)
        reward_layout.addWidget(QLabel("<b>Penalties:</b>"), 0, 2, 1, 2)
        
        reward_layout.addWidget(QLabel("Rheotaxis:"), 1, 2)
        self.rheotaxis_penalty_spin = QDoubleSpinBox()
        self.rheotaxis_penalty_spin.setRange(-100, 100)
        self.rheotaxis_penalty_spin.setValue(-1.0)
        self.rheotaxis_penalty_spin.setDecimals(2)
        self.rheotaxis_penalty_spin.setToolTip("Penalty for facing wrong direction")
        reward_layout.addWidget(self.rheotaxis_penalty_spin, 1, 3)
        
        reward_layout.addWidget(QLabel("Separation:"), 2, 2)
        self.separation_penalty_spin = QDoubleSpinBox()
        self.separation_penalty_spin.setRange(-100, 100)
        self.separation_penalty_spin.setValue(0.005)
        self.separation_penalty_spin.setDecimals(4)
        self.separation_penalty_spin.setToolTip("Penalty for crowding violations")
        reward_layout.addWidget(self.separation_penalty_spin, 2, 3)
        
        reward_layout.addWidget(QLabel("Mortality:"), 3, 2)
        self.mortality_penalty_spin = QDoubleSpinBox()
        self.mortality_penalty_spin.setRange(-100, 100)
        self.mortality_penalty_spin.setValue(-50.0)
        self.mortality_penalty_spin.setDecimals(1)
        self.mortality_penalty_spin.setToolTip("Penalty per fish death")
        reward_layout.addWidget(self.mortality_penalty_spin, 3, 3)
        
        reward_layout.addWidget(QLabel("Fatigue:"), 4, 2)
        self.fatigue_penalty_spin = QDoubleSpinBox()
        self.fatigue_penalty_spin.setRange(-100, 100)
        self.fatigue_penalty_spin.setValue(-0.1)
        self.fatigue_penalty_spin.setDecimals(3)
        self.fatigue_penalty_spin.setToolTip("Penalty for low battery timesteps")
        reward_layout.addWidget(self.fatigue_penalty_spin, 4, 3)
        
        reward_layout.addWidget(QLabel("Stagnation:"), 5, 2)
        self.stagnation_penalty_spin = QDoubleSpinBox()
        self.stagnation_penalty_spin.setRange(-100, 100)
        self.stagnation_penalty_spin.setValue(-0.2)
        self.stagnation_penalty_spin.setDecimals(3)
        self.stagnation_penalty_spin.setToolTip("Penalty for stationary timesteps")
        reward_layout.addWidget(self.stagnation_penalty_spin, 5, 3)
        
        # Column 3: More penalties
        reward_layout.addWidget(QLabel("<b>Other:</b>"), 0, 4, 1, 2)
        
        reward_layout.addWidget(QLabel("Smoothness:"), 1, 4)
        self.smoothness_penalty_spin = QDoubleSpinBox()
        self.smoothness_penalty_spin.setRange(-100, 100)
        self.smoothness_penalty_spin.setValue(-0.001)
        self.smoothness_penalty_spin.setDecimals(4)
        self.smoothness_penalty_spin.setToolTip("Penalty for jerky movement (jerk sum)")
        reward_layout.addWidget(self.smoothness_penalty_spin, 1, 5)
        
        reward_layout.addWidget(QLabel("Boundary:"), 2, 4)
        self.boundary_penalty_spin = QDoubleSpinBox()
        self.boundary_penalty_spin.setRange(-100, 100)
        self.boundary_penalty_spin.setValue(0.0)
        self.boundary_penalty_spin.setDecimals(2)
        self.boundary_penalty_spin.setToolTip("Penalty for boundary proximity (disabled)")
        reward_layout.addWidget(self.boundary_penalty_spin, 2, 5)
        
        reward_group.setLayout(reward_layout)
        reward_group.setToolTip("These weights define WHAT YOU VALUE (objective function). They are NOT trained by RL.")
        layout.addWidget(reward_group)
        
        # Randomize buttons
        self.btn_randomize = QPushButton("🎲 Randomize Weights")
        self.btn_randomize.clicked.connect(self.randomize_weights.emit)
        self.btn_randomize.setToolTip("Generate new random initial weights (100% variation from defaults). Only works before training starts.")
        layout.addWidget(self.btn_randomize)
        
        self.btn_randomize_order = QPushButton("🎪 Randomize Cue Order")
        self.btn_randomize_order.clicked.connect(self.randomize_order.emit)
        self.btn_randomize_order.setToolTip("Shuffle behavioral cue application order. Order stays fixed throughout all episodes. Only works before training starts.")
        layout.addWidget(self.btn_randomize_order)
        
        # Reset button
        self.btn_reset = QPushButton("🔄 Reset Training")
        self.btn_reset.clicked.connect(self.reset_training.emit)
        self.btn_reset.setToolTip("Reset to initial state (clears training history, keeps current weights). Start fresh without restarting the app.")
        layout.addWidget(self.btn_reset)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Training progress plot
        plot_group = QGroupBox("Training Progress")
        plot_layout = QVBoxLayout()
        
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            
            self.figure = Figure(figsize=(5, 3), dpi=80)
            self.canvas = FigureCanvasQTAgg(self.figure)
            
            # Set minimum size to prevent negative dimensions
            self.canvas.setMinimumSize(200, 150)
            
            # Disable matplotlib's default scroll/pan/zoom navigation toolbar behavior
            # We need to disconnect the NavigationToolbar2QT scroll handler
            try:
                # Disconnect all scroll_event callbacks to prevent pan/zoom interference
                callbacks = self.canvas.callbacks.callbacks.get('scroll_event', {})
                for cid in list(callbacks.keys()):
                    self.canvas.mpl_disconnect(cid)
            except Exception:
                pass  # If no callbacks exist, that's fine
            
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.set_title('RL Training Progress')
            self.ax.grid(True, alpha=0.3)
            self.figure.tight_layout()
            
            plot_layout.addWidget(self.canvas)
            self.has_plot = True
        except ImportError:
            # Fallback to text log if matplotlib not available
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setMaximumHeight(200)
            self.log_text.setFont(QFont("Courier New", 8))
            plot_layout.addWidget(self.log_text)
            self.has_plot = False
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # Training history for plotting
        self.episode_history = []
        self.reward_history = []
        
        layout.addStretch()
        self.setLayout(layout)
    
    def _on_episode_combo_changed(self, index):
        """Handle episode selection from dropdown."""
        if index >= 0:
            self.episode_selected.emit(index)
    
    def add_episode_to_list(self, episode_num: int):
        """Add completed episode to navigation dropdown."""
        self.episode_combo.addItem(f"Episode {episode_num + 1}")
        # Auto-select newest episode
        self.episode_combo.setCurrentIndex(self.episode_combo.count() - 1)
    
    def append_log(self, message: str):
        """Append message to training log (if using text log fallback)."""
        if not self.has_plot:
            self.log_text.append(message)
            # Auto-scroll to bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
    
    def update_plot(self, episode: int, reward: float):
        """Update training progress plot."""
        if not self.has_plot:
            return
        
        self.episode_history.append(episode)
        self.reward_history.append(reward)
        
        self.ax.clear()
        self.ax.plot(self.episode_history, self.reward_history, 'b-', linewidth=2, label='Episode Reward')
        
        # Mark best reward
        if len(self.reward_history) > 0:
            best_idx = np.argmax(self.reward_history)
            self.ax.plot(self.episode_history[best_idx], self.reward_history[best_idx], 
                        'r*', markersize=12, label='Best')
        
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.set_title('RL Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def set_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
        else:
            self.progress_bar.setValue(0)
    
    def set_parameters_enabled(self, enabled: bool):
        """Enable or disable parameter spinboxes."""
        self.episodes_spin.setEnabled(enabled)
        self.timesteps_spin.setEnabled(enabled)
        self.agents_spin.setEnabled(enabled)
        self.noise_spin.setEnabled(enabled)
        self.btn_randomize.setEnabled(enabled)
        self.btn_randomize.setEnabled(enabled)


class TrainingWorker(QObject):
    """
    Background worker for running RL training without blocking UI.
    """
    
    # Signals
    episode_started = pyqtSignal(int)  # episode number
    episode_computed = pyqtSignal(int, float, dict, object, object, object, object, object)  # episode, reward, components, positions, headings, battery, alive, weights
    training_completed = pyqtSignal(object, list)  # best_weights, history
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, trainer, num_episodes: int):
        super().__init__()
        self.trainer = trainer
        self.num_episodes = num_episodes
        self.is_stopped = False
        self.is_paused = False
        
        # Synchronization: wait for animation to complete
        self.animation_complete = threading.Event()
        self.animation_complete.set()  # Initially ready
        
        # Cache longitudinal profile (create sim once to get it)
        print("Caching longitudinal profile from simulation...", flush=True)
        self.longitudinal_profile = None
        try:
            sim = self.trainer.simulation_factory(self.trainer.initial_weights)
            self.longitudinal_profile = getattr(sim, 'longitudinal', None)
            print(f"Simulation created, checking longitudinal attribute...", flush=True)
            if self.longitudinal_profile is not None:
                print(f"Longitudinal profile cached: {type(self.longitudinal_profile)}", flush=True)
            else:
                print(f"WARNING: sim.longitudinal is None", flush=True)
            sim.close()
        except Exception as e:
            import traceback
            error_msg = f"ERROR caching longitudinal profile: {e}\n{traceback.format_exc()}"
            print(error_msg, flush=True)
            raise RuntimeError(f"Failed to cache longitudinal profile: {e}") from e
        
        if self.longitudinal_profile is None:
            raise ValueError(
                "Longitudinal profile not loaded from simulation.\n"
                "Check that longitudinal_profile path is correct and shapefile loads properly.\n"
                "Simulation may have failed to import the shapefile."
            )
    

    def run(self):
        """Execute training loop with UI updates."""
        try:
            current_weights = self.trainer.initial_weights
            
            for episode in range(self.num_episodes):
                # Check for stop signal
                if self.is_stopped:
                    break
                    
                # Check for pause signal
                while self.is_paused and not self.is_stopped:
                    time.sleep(0.1)
                    
                if self.is_stopped:
                    break
                
                self.episode_started.emit(episode)
                
                # Run episode - this executes ALL timesteps before returning
                positions, headings, velocities, battery, alive, velocity_field = self.trainer.run_episode(current_weights)
                
                # Log completion with actual timestep count
                actual_timesteps = positions.shape[0]
                print(f"Episode {episode}: Completed {actual_timesteps} timesteps")
                
                # Compute reward (use flow vector integration for upstream progress)
                from emergent.salmon_abm.rl_training import compute_episode_reward
                # Pass current weights to reward function for constraint checking
                reward, components = compute_episode_reward(
                    positions, headings, velocities, alive,
                    body_length=self.trainer.body_length,
                    threat_level=current_weights.threat_level,
                    behavioral_weights=current_weights.to_dict(),
                    battery_history=battery,
                    longitudinal_profile=self.longitudinal_profile,
                    velocity_field_history=velocity_field  # Flow integration for braided channels
                )
                
                # Track best
                if reward > self.trainer.best_reward:
                    self.trainer.best_reward = reward
                    self.trainer.best_weights = current_weights
                    
                # Store history
                self.trainer.episode_history.append((episode, float(reward)))
                
                # Emit episode data immediately - viewer will queue if busy
                # NO WAITING - next episode computes while current animates!
                self.episode_computed.emit(episode, float(reward), components, positions, headings, battery, alive, current_weights)
                
                # Mutate for next episode (compute in parallel with visualization)
                current_weights = self.trainer.best_weights.mutate(
                    mutation_scale=self.trainer.exploration_noise
                )
            
            # Training complete
            self.training_completed.emit(self.trainer.best_weights, self.trainer.episode_history)
            
        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")
    
    def pause(self):
        """Pause training."""
        self.is_paused = True
        
    def resume(self):
        """Resume training."""
        self.is_paused = False
        
    def stop(self):
        """Stop training."""
        self.is_stopped = True


class RLTrainingViewer(QMainWindow):
    """
    Main window for RL training visualization.
    
    Three-panel layout:
    - Left: Behavioral weights and diagnostics
    - Center: Simulation visualization
    - Right: Training controls
    """
    
    def __init__(self, model_dir: Optional[str] = None, start_polygon: Optional[str] = None,
                 model_name: str = "salmon_abm", basin: str = "nuyakuk"):
        super().__init__()
        self.setWindowTitle("RL Training Visualizer - Behavioral Weight Optimization")
        self.resize(1400, 800)
        
        # Configuration
        self.model_dir = model_dir
        self.start_polygon = start_polygon
        self.model_name = model_name
        self.basin = basin
        self.crs = "EPSG:26905"  # UTM Zone 5N for Alaska
        self.water_temp = 12.0  # °C
        
        # Training state
        self.trainer = None
        self.training_thread = None
        self.training_worker = None
        self.training_process = None
        self.training_queue = None
        self.training_control_queue = None
        self.training_poll_timer = None
        self.training_paused = False
        self.initial_reward = None
        
        # Episode visualization queue (for parallel computation)
        from collections import deque
        self.episode_queue = deque()  # Queue of (episode, reward, components, positions, headings, battery, alive, weights)
        self.is_animating = False  # Track if viewer is currently animating
        
        # Completed episodes storage (for replay)
        self.completed_episodes = []  # List of (episode, reward, components, positions, headings, battery, alive, weights)
        self.best_episode_data = None  # Always keep best episode: (episode, reward, components, positions, headings, battery, alive, weights)
        self.best_reward = float('-inf')  # Track best reward seen
        self.best_weights = None
        
        # Current behavioral weights (modified by randomize buttons)
        from emergent.salmon_abm.rl_training import BehavioralWeights
        self.current_weights = BehavioralWeights()
        
        # Episode synchronization
        self.pending_episode_data = None  # Stores (episode, reward, components) waiting for animation
        
        print(f"RLTrainingViewer init: calling init_ui()...", flush=True)
        try:
            self.init_ui()
            print(f"RLTrainingViewer init: init_ui() complete", flush=True)
        except Exception as e:
            import traceback
            print(f"ERROR in init_ui: {e}", flush=True)
            traceback.print_exc()
            raise
        
    def init_ui(self):
        """Initialize UI components."""
        # Create panels
        self.weights_panel = WeightsPanel()
        self.simulation_canvas = SimulationCanvas(model_dir=self.model_dir)
        self.control_panel = ControlPanel()
        
        # Connect control signals
        self.control_panel.start_training.connect(self.on_start_training)
        self.control_panel.pause_training.connect(self.on_pause_training)
        self.control_panel.stop_training.connect(self.on_stop_training)
        self.control_panel.randomize_weights.connect(self.on_randomize_weights)
        self.control_panel.set_blanket_value.connect(self.on_set_blanket_value)
        self.control_panel.randomize_order.connect(self.on_randomize_order)
        self.control_panel.reset_training.connect(self.on_reset_training)
        self.control_panel.episode_selected.connect(self.on_episode_selected)
        self.control_panel.next_episode.connect(self.on_next_episode)
        self.control_panel.prev_episode.connect(self.on_prev_episode)
        self.control_panel.toggle_playback.connect(self.on_toggle_playback)
        self.control_panel.restart_animation.connect(self.on_restart_animation)
        
        # Connect animation finished signal
        self.simulation_canvas.animation_finished.connect(self.on_animation_finished)
        
        # Create splitter for three panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.weights_panel)
        splitter.addWidget(self.simulation_canvas)
        splitter.addWidget(self.control_panel)
        
        # Set relative sizes (20% left, 50% center, 30% right)
        splitter.setSizes([280, 700, 420])
        
        self.setCentralWidget(splitter)
        
        # Display initial weights
        self.weights_panel.update_weights(self.current_weights)

    def _is_training_active(self) -> bool:
        return self.training_process is not None and self.training_process.is_alive()

    def _start_training_poll(self) -> None:
        if self.training_poll_timer is None:
            self.training_poll_timer = QTimer(self)
            self.training_poll_timer.setInterval(100)
            self.training_poll_timer.timeout.connect(self._poll_training_queue)
        if not self.training_poll_timer.isActive():
            self.training_poll_timer.start()

    def _stop_training_poll(self) -> None:
        if self.training_poll_timer is not None and self.training_poll_timer.isActive():
            self.training_poll_timer.stop()

    def _poll_training_queue(self) -> None:
        if self.training_queue is None:
            return
        handled = 0
        while handled < 20:
            try:
                msg = self.training_queue.get_nowait()
            except queue_mod.Empty:
                break
            self._handle_training_message(msg)
            handled += 1
        if self.training_process is not None and not self.training_process.is_alive():
            if self.training_queue is None or self.training_queue.empty():
                self._cleanup_training_process()

    def _handle_training_message(self, msg: Dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type == "episode_started":
            self.on_episode_started(int(msg.get("episode", 0)))
            return
        if msg_type == "episode_computed":
            weights_dict = msg.get("weights") or {}
            current_weights = BehavioralWeights.from_dict(weights_dict) if weights_dict else None
            reward = float(msg.get("reward", 0.0))
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_weights = current_weights
            self.on_episode_computed(
                int(msg.get("episode", 0)),
                reward,
                msg.get("components", {}),
                msg.get("positions"),
                msg.get("headings"),
                msg.get("battery"),
                msg.get("alive"),
                current_weights,
            )
            return
        if msg_type == "training_completed":
            best_weights = BehavioralWeights.from_dict(msg.get("best_weights") or {})
            self.best_weights = best_weights
            history = msg.get("history", [])
            self.on_training_completed(best_weights, history)
            self._cleanup_training_process()
            return
        if msg_type == "training_stopped":
            best_weights = BehavioralWeights.from_dict(msg.get("best_weights") or {})
            self.best_weights = best_weights
            self._cleanup_training_process()
            self.control_panel.append_log("Training stopped")
            self.control_panel.status_label.setText("Stopped")
            self.control_panel.set_parameters_enabled(True)
            return
        if msg_type == "error":
            self.on_error_occurred(msg.get("message", "Unknown training error"))
            self._cleanup_training_process()
            return

    def _cleanup_training_process(self) -> None:
        self._stop_training_poll()
        if self.training_process is not None and self.training_process.is_alive():
            self.training_process.join(timeout=1.0)
        self.training_process = None
        self.training_queue = None
        self.training_control_queue = None
        self.training_paused = False
        self.training_thread = None
        self.training_worker = None
        
    def create_simulation_factory(self, num_agents: int, num_timesteps: int):
        """
        Create simulation factory function for RL training.
        
        Args:
            num_agents: Number of agents per episode
            num_timesteps: Number of timesteps per episode
            
        Returns:
            Callable that takes BehavioralWeights and returns a configured simulation object
        """
        # Find environment files
        env_files = []
        for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']:
            fpath = os.path.join(self.model_dir, fname)
            if os.path.exists(fpath):
                env_files.append(fpath)
        
        if not env_files:
            raise FileNotFoundError(f"No environment files found in {self.model_dir}")
        
        self.control_panel.append_log(f"Found {len(env_files)} environment files")
        
        # Look for longitudinal profile shapefile (REQUIRED for reward function)
        longitudinal_path = os.path.join(self.model_dir, 'longitudinal.shp')
        if not os.path.exists(longitudinal_path):
            raise FileNotFoundError(
                f"Longitudinal profile shapefile required but not found: {longitudinal_path}\n"
                f"The reward function requires this to compute upstream progress accurately."
            )
        self.control_panel.append_log(f"Found longitudinal profile: {longitudinal_path}")
        
        def factory_func(weights: BehavioralWeights):
            """Create and configure simulation with given behavioral weights."""
            # Create simulation with minimal output writes (compute-only)
            sim = simulation(
                model_dir=self.model_dir,
                model_name=self.model_name,
                crs=self.crs,
                basin=self.basin,
                water_temp=self.water_temp,
                start_polygon=self.start_polygon,
                env_files=env_files,
                longitudinal_profile=longitudinal_path,
                fish_length=None,  # Random fish lengths
                num_timesteps=num_timesteps,
                num_agents=num_agents,
                use_gpu=False,
                pid_tuning=False,
                db_path=None,  # temporary DB (auto-cleanup)
                output_write_mode='none',  # skip all writes (compute-only)
                output_write_backend='sync'
            )
            
            # Load behavioral weights into simulation
            sim.load_behavioral_weights(weights_dict=weights.to_dict())
            
            # Enable neighbor sensing for schooling cues
            fish_length_m = 0.3  # Approximate 300mm fish
            sensory_range = 5.0  # Fixed biological constant (5 BL = 1.5m)
            sim.neighbor_buffer_radius = sensory_range * fish_length_m
            sim.neighbor_buffer_lengths = sensory_range
            
            # RANDOMIZE initial conditions for RL exploration
            # Random headings [0, 2π] instead of upstream direction
            sim.heading = np.random.uniform(0, 2*np.pi, sim.num_agents).astype(np.float32)
            # Random initial SOG [0.1, 1.5] m/s instead of ideal_sog
            sim.sog = np.random.uniform(0.1, 1.5, sim.num_agents).astype(np.float32)
            
            return sim
        
        return factory_func
        
    def on_start_training(self):
        """Start training with current parameters."""
        if self._is_training_active():
            self.control_panel.append_log("Training already in progress!")
            return
        
        # Check configuration
        if not self.model_dir or not self.start_polygon:
            self.control_panel.append_log("ERROR: Missing configuration!")
            self.control_panel.append_log("Please provide --model-dir and --start-polygon arguments")
            return
        
        if not os.path.exists(self.model_dir):
            self.control_panel.append_log(f"ERROR: Model directory not found: {self.model_dir}")
            return
            
        if not os.path.exists(self.start_polygon):
            self.control_panel.append_log(f"ERROR: Start polygon not found: {self.start_polygon}")
            return
            
        # Get parameters from control panel
        num_episodes = self.control_panel.episodes_spin.value()
        num_timesteps = self.control_panel.timesteps_spin.value()
        num_agents = self.control_panel.agents_spin.value()
        exploration_noise = self.control_panel.noise_spin.value()
        
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log(f"Starting RL training:")
        self.control_panel.append_log(f"  Episodes: {num_episodes}")
        self.control_panel.append_log(f"  Timesteps: {num_timesteps}")
        self.control_panel.append_log(f"  Agents: {num_agents}")
        self.control_panel.append_log(f"  Exploration: {exploration_noise}")
        self.control_panel.status_label.setText("Initializing...")
        
        try:
            # Resolve environment files for subprocess
            env_files = []
            for fname in ['depth.tif', 'vel_x.tif', 'vel_y.tif', 'vel_mag.tif', 'vel_dir.tif']:
                fpath = os.path.join(self.model_dir, fname)
                if os.path.exists(fpath):
                    env_files.append(fpath)
            if not env_files:
                raise FileNotFoundError(f"No environment files found in {self.model_dir}")
            self.control_panel.append_log(f"Found {len(env_files)} environment files")

            longitudinal_path = os.path.join(self.model_dir, 'longitudinal.shp')
            if not os.path.exists(longitudinal_path):
                raise FileNotFoundError(
                    f"Longitudinal profile shapefile required but not found: {longitudinal_path}\n"
                    f"The reward function requires this to compute upstream progress accurately."
                )
            self.control_panel.append_log(f"Found longitudinal profile: {longitudinal_path}")

            # Get edited weights from the panel (user may have manually edited values)
            initial_weights = self.weights_panel.get_edited_weights(self.current_weights)
            self.current_weights = initial_weights  # Update current_weights with edits

            reward_weights = {
                'cohesion': self.control_panel.cohesion_reward_spin.value(),
                'alignment': self.control_panel.alignment_reward_spin.value(),
                'separation': self.control_panel.separation_penalty_spin.value(),
                'upstream_progress': self.control_panel.upstream_weight_spin.value(),
                'energy_efficiency': self.control_panel.energy_reward_spin.value(),
                'drafting_benefit': self.control_panel.drafting_reward_spin.value(),
                'boundary_penalty': self.control_panel.boundary_penalty_spin.value(),
                'mortality_penalty': self.control_panel.mortality_penalty_spin.value(),
                'smoothness_penalty': self.control_panel.smoothness_penalty_spin.value(),
                'fatigue_penalty': self.control_panel.fatigue_penalty_spin.value(),
                'stagnation_penalty': self.control_panel.stagnation_penalty_spin.value(),
                'rheotaxis_alignment': self.control_panel.rheotaxis_penalty_spin.value(),
            }

            config = {
                "model_dir": self.model_dir,
                "model_name": self.model_name,
                "crs": self.crs,
                "basin": self.basin,
                "water_temp": self.water_temp,
                "start_polygon": self.start_polygon,
                "env_files": env_files,
                "longitudinal_path": longitudinal_path,
                "num_episodes": num_episodes,
                "num_timesteps": num_timesteps,
                "num_agents": num_agents,
                "exploration_noise": exploration_noise,
                "reward_weights": reward_weights,
                "body_length": 0.3,
                "dt": 1.0,
                "initial_weights": initial_weights.to_dict(),
            }

            self.best_reward = float('-inf')
            self.best_weights = None
            self.initial_reward = None
            self.training_paused = False

            self.control_panel.append_log("Trainer initialized successfully (subprocess)")

            # Display initial weights
            self.weights_panel.update_weights(initial_weights)

            default_order = {
                0: 'shallow',
                1: 'border',
                2: 'avoid',
                3: 'collision',
                4: 'alignment',
                5: 'cohesion',
                6: 'low_speed',
                7: 'refugia',
                8: 'rheotaxis',
                9: 'wave_drag',
            }
            self.weights_panel.update_order(default_order, initial_weights)

            # Disable parameter controls during training
            self.control_panel.set_parameters_enabled(False)

            ctx = mp.get_context("spawn")
            self.training_queue = ctx.Queue(maxsize=2)
            self.training_control_queue = ctx.Queue()
            self.training_process = ctx.Process(
                target=_training_process_main,
                args=(config, self.training_queue, self.training_control_queue),
            )
            self.training_process.start()
            self._start_training_poll()
            self.control_panel.append_log("Training process started")

        except Exception as e:
            import traceback
            self.control_panel.append_log(f"ERROR: Failed to start training")
            self.control_panel.append_log(str(e))
            self.control_panel.append_log(traceback.format_exc())
            self.control_panel.status_label.setText("Error")
    
    def on_set_blanket_value(self):
        """Set all behavioral weights to a uniform blanket value."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot set blanket value during training")
            return
        
        # Get blanket value from input field
        try:
            blanket_value = float(self.control_panel.blanket_value_input.text())
        except ValueError:
            self.control_panel.append_log("ERROR: Invalid blanket value (must be a number)")
            return
        
        # Directly modify current weights - don't create new instance
        # (creating new BehavioralWeights() resets to defaults)
        self.current_weights.shallow_weight = blanket_value
        self.current_weights.border_cue_weight = blanket_value
        self.current_weights.avoid_weight = blanket_value
        self.current_weights.collision_weight = blanket_value
        self.current_weights.alignment_weight = blanket_value
        self.current_weights.cohesion_weight = blanket_value
        self.current_weights.low_speed_weight = blanket_value
        self.current_weights.refugia_weight = blanket_value
        self.current_weights.rheotaxis_weight = blanket_value
        self.current_weights.wave_drag_weight = blanket_value
        
        # Debug: confirm values were set
        print(f"[BLANKET VALUE DEBUG] Set all weights to {blanket_value}", flush=True)
        print(f"[BLANKET VALUE DEBUG] shallow_weight={self.current_weights.shallow_weight}, cohesion_weight={self.current_weights.cohesion_weight}", flush=True)
        
        # Update spinbox values directly (don't rebuild UI)
        weight_keys = [
            'shallow_weight', 'border_cue_weight', 'avoid_weight', 'collision_weight',
            'alignment_weight', 'cohesion_weight', 'low_speed_weight', 'refugia_weight',
            'rheotaxis_weight', 'wave_drag_weight'
        ]
        for key in weight_keys:
            if key in self.weights_panel.weight_spinboxes:
                self.weights_panel.weight_spinboxes[key].setValue(blanket_value)
        
        self.control_panel.append_log(f"📋 Set all weights to blanket value: {blanket_value}")
        self.control_panel.status_label.setText(f"Ready with uniform weights ({blanket_value})")
    
    def on_randomize_weights(self):
        """Regenerate random initial weights."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot randomize during training")
            return
        
        # Preserve current order before randomizing
        old_order = [
            self.current_weights.order_0, self.current_weights.order_1,
            self.current_weights.order_2, self.current_weights.order_3,
            self.current_weights.order_4, self.current_weights.order_5,
            self.current_weights.order_6, self.current_weights.order_7,
            self.current_weights.order_8, self.current_weights.order_9
        ]
        
        # Generate new randomized weights
        from emergent.salmon_abm.rl_training import BehavioralWeights
        base_weights = BehavioralWeights()
        new_weights = base_weights.randomize(scale=1.0)  # 100% randomization
        
        # Restore order (as integers)
        new_weights.order_0 = int(old_order[0])
        new_weights.order_1 = int(old_order[1])
        new_weights.order_2 = int(old_order[2])
        new_weights.order_3 = int(old_order[3])
        new_weights.order_4 = int(old_order[4])
        new_weights.order_5 = int(old_order[5])
        new_weights.order_6 = int(old_order[6])
        new_weights.order_7 = int(old_order[7])
        new_weights.order_8 = int(old_order[8])
        new_weights.order_9 = int(old_order[9])
        
        self.current_weights = new_weights  # Store as current for training
        
        # Update display
        self.weights_panel.update_weights(new_weights)
        
        # Update order display (need default_order dict)
        default_order = {
            0: 'shallow', 1: 'border', 2: 'avoid', 3: 'collision', 4: 'alignment',
            5: 'cohesion', 6: 'low_speed', 7: 'refugia', 8: 'rheotaxis', 9: 'wave_drag'
        }
        self.weights_panel.update_order(default_order, new_weights)
        
        self.control_panel.append_log("🎲 Randomized initial weights (100% variation)")
        self.control_panel.status_label.setText("Ready with new random weights")
    
    def on_randomize_order(self):
        """Shuffle cue application order."""
        if self._is_training_active():
            self.control_panel.append_log("Cannot randomize order during training")
            return
        
        # Random permutation of [0,1,2,3,4,5,6,7,8,9]
        new_order = np.random.permutation(10)
        
        # Update current_weights with shuffled order (as integers)
        self.current_weights.order_0 = int(new_order[0])
        self.current_weights.order_1 = int(new_order[1])
        self.current_weights.order_2 = int(new_order[2])
        self.current_weights.order_3 = int(new_order[3])
        self.current_weights.order_4 = int(new_order[4])
        self.current_weights.order_5 = int(new_order[5])
        self.current_weights.order_6 = int(new_order[6])
        self.current_weights.order_7 = int(new_order[7])
        self.current_weights.order_8 = int(new_order[8])
        self.current_weights.order_9 = int(new_order[9])
        
        # Default cue names for display
        default_order = {
            0: 'shallow', 1: 'border', 2: 'avoid', 3: 'collision', 4: 'alignment',
            5: 'cohesion', 6: 'low_speed', 7: 'refugia', 8: 'rheotaxis', 9: 'wave_drag'
        }
        
        # Update display with remapping
        self.weights_panel.update_order(default_order, self.current_weights)
        self.control_panel.append_log(f"🎪 Randomized cue order: {new_order}")
        self.control_panel.status_label.setText("Ready with shuffled cue order")
    
    def on_reset_training(self):
        """Reset training to initial state - stop all processes and start fresh."""
        # Stop training if running
        if self._is_training_active():
            self.control_panel.append_log("Stopping training for reset...")
            if self.training_control_queue is not None:
                self.training_control_queue.put("stop")
            if self.training_process is not None:
                self.training_process.join(timeout=2.0)
                if self.training_process.is_alive():
                    self.training_process.terminate()
            self._cleanup_training_process()
        
        # Clear episode data
        self.episode_queue.clear()
        self.completed_episodes.clear()
        self.is_animating = False
        self.pending_episode_data = None
        self._current_episode_data = None
        
        # Clear visualization
        if hasattr(self.simulation_canvas, 'replay_widget'):
            self.simulation_canvas.replay_widget.positions = None
            self.simulation_canvas.replay_widget.playing = False
            self.simulation_canvas.replay_widget.timer.stop()
            self.simulation_canvas.replay_widget.update()
        
        num_agents = self.control_panel.agents_spin.value()
        num_timesteps = self.control_panel.timesteps_spin.value()
        self.trainer = None
        self.training_thread = None
        self.training_worker = None
        self.initial_reward = None
        self.best_reward = float('-inf')
        self.best_weights = None
        
        # Clear plot
        if hasattr(self.control_panel, 'has_plot') and self.control_panel.has_plot:
            try:
                self.control_panel.ax.clear()
                self.control_panel.ax.set_xlabel('Episode')
                self.control_panel.ax.set_ylabel('Reward')
                self.control_panel.ax.set_title('RL Training Progress')
                self.control_panel.ax.grid(True, alpha=0.3)
                self.control_panel.canvas.draw()
            except Exception:
                pass
        
        # Reset UI
        self.control_panel.progress_bar.setValue(0)
        self.control_panel.btn_pause.setText("⏸ Pause Training")
        self.control_panel.btn_play_pause.setText("⏸ Pause")
        self.control_panel.status_label.setText("Ready (Reset)")
        self.weights_panel.clear_diagnostics()
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log("Training reset - all processes stopped, new model created")
        self.control_panel.append_log(f"Ready for new training run ({num_agents} agents, {num_timesteps} timesteps)")
        self.control_panel.append_log("=" * 50)
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
        
        self.control_panel.append_log("=" * 50)
        self.control_panel.append_log("🔄 Training reset - ready to start fresh")
        self.control_panel.append_log("Current weights preserved - press Randomize for new weights")
        
    def on_pause_training(self):
        """Pause/resume training."""
        if not self._is_training_active():
            return
        if self.training_control_queue is None:
            return
        if self.training_paused:
            self.training_control_queue.put("resume")
            self.training_paused = False
            self.control_panel.btn_pause.setText("⏸ Pause Training")
            self.control_panel.append_log("Training resumed")
        else:
            self.training_control_queue.put("pause")
            self.training_paused = True
            self.control_panel.btn_pause.setText("▶ Resume Training")
            self.control_panel.append_log("Training paused")
    
    def on_toggle_playback(self):
        """Toggle play/pause for current episode animation."""
        if hasattr(self.simulation_canvas, 'replay_widget'):
            if self.simulation_canvas.replay_widget.playing:
                self.simulation_canvas.replay_widget.playing = False
                self.simulation_canvas.replay_widget.timer.stop()
                self.control_panel.btn_play_pause.setText("▶ Play")
                self.control_panel.append_log("Animation paused")
            else:
                self.simulation_canvas.replay_widget.playing = True
                self.simulation_canvas.replay_widget.timer.start()
                self.control_panel.btn_play_pause.setText("⏸ Pause")
                self.control_panel.append_log("Animation playing")
    
    def on_restart_animation(self):
        """Restart current episode animation from beginning."""
        if hasattr(self.simulation_canvas, 'replay_widget'):
            self.simulation_canvas.replay_widget.frame = 0
            self.simulation_canvas.replay_widget.playing = True
            self.simulation_canvas.replay_widget.timer.start()
            self.control_panel.btn_play_pause.setText("⏸ Pause")
            self.simulation_canvas.replay_widget.update()
            self.control_panel.append_log("Animation restarted")
                
    def on_stop_training(self):
        """Stop training."""
        if self._is_training_active():
            if self.training_control_queue is not None:
                self.training_control_queue.put("stop")
            self.control_panel.append_log("Stopping training...")
            self.control_panel.status_label.setText("Stopping...")
            self.control_panel.set_parameters_enabled(True)
            
    def on_episode_started(self, episode: int):
        """Handle episode start."""
        total = self.control_panel.episodes_spin.value()
        self.control_panel.set_progress(episode, total)
        self.control_panel.status_label.setText(f"Computing episode {episode + 1}/{total}...")
        
    def on_episode_computed(self, episode: int, reward: float, components: Dict[str, float], 
                           positions: np.ndarray, headings: np.ndarray, battery: np.ndarray, alive: np.ndarray, weights):
        """Handle episode computation complete - queue for visualization."""
        # Add episode to queue
        self.episode_queue.append((episode, reward, components, positions, headings, battery, alive, weights))
        
        # Update status
        total = self.control_panel.episodes_spin.value()
        queue_len = len(self.episode_queue)
        self.control_panel.status_label.setText(f"Computed episode {episode + 1}/{total} (queue: {queue_len})")
        
        # If not currently animating, start processing queue
        if not self.is_animating:
            self.process_next_queued_episode()
    
    def process_next_queued_episode(self):
        """Process next episode from queue."""
        print(f"[RL VIEWER DEBUG] process_next_queued_episode called, queue length={len(self.episode_queue)}", flush=True)
        if len(self.episode_queue) == 0:
            print("[RL VIEWER DEBUG] Queue empty, returning", flush=True)
            return
        
        # Mark as animating
        self.is_animating = True
        print("[RL VIEWER DEBUG] Marked as animating", flush=True)
        
        # Get next episode from queue
        episode, reward, components, positions, headings, battery, alive, weights = self.episode_queue.popleft()
        print(f"[RL VIEWER DEBUG] Got episode {episode} from queue, positions shape={positions.shape}", flush=True)
        
        # Store full episode data for later saving to completed_episodes
        self._current_episode_data = (episode, reward, components, positions.copy(), headings.copy(), battery.copy(), alive.copy(), weights)
        
        # Store for processing after animation
        self.pending_episode_data = (episode, reward, components, weights)
        
        # Update status
        total = self.control_panel.episodes_spin.value()
        queue_len = len(self.episode_queue)
        self.control_panel.status_label.setText(f"Visualizing episode {episode + 1}/{total} (queue: {queue_len})")
        
        # Start visualization - this will trigger animation_finished when done
        print(f"[RL VIEWER DEBUG] Calling set_positions for episode {episode}", flush=True)
        self.simulation_canvas.set_positions(positions, headings, battery, alive)
        
    def on_animation_finished(self):
        """Handle animation playback complete - update UI and process next queued episode."""
        if self.pending_episode_data is None:
            self.is_animating = False
            return
            
        episode, reward, components, current_weights = self.pending_episode_data
        
        # Diagnostic logging: Track weight evolution and warn about collapse
        if self.best_weights is not None:
            weights_dict = self.best_weights.to_dict()
            cohesion_w = weights_dict.get('cohesion', 0)
            alignment_w = weights_dict.get('alignment', 0)
            rheotaxis_w = weights_dict.get('rheotaxis', 0)
            collision_w = weights_dict.get('collision', 0)
            refugia_w = weights_dict.get('refugia', 0)
            
            weight_summary = f"Rheo:{rheotaxis_w:.0f} Coh:{cohesion_w:.0f} Align:{alignment_w:.0f} Coll:{collision_w:.0f} Ref:{refugia_w:.0f}"
            
            # CRITICAL WARNING if schooling weights drop too low
            if cohesion_w < 1000 or alignment_w < 1000 or collision_w < 1000:
                self.control_panel.append_log(
                    f"⚠️  WARNING Ep{episode+1}: Schooling collapse! {weight_summary}"
                )
        else:
            weight_summary = "(weights unavailable)"
        self.pending_episode_data = None
        
        total = self.control_panel.episodes_spin.value()
        
        # Update diagnostics
        if self.initial_reward is None:
            self.initial_reward = reward
            
        best_reward = self.best_reward if self.best_reward != float('-inf') else reward
        self.weights_panel.update_diagnostics(episode + 1, total, reward, best_reward, self.initial_reward)
        self.weights_panel.update_components(components)
        
        # Update weights display with CURRENT episode's weights (not just best)
        if current_weights:
            self.weights_panel.update_weights(current_weights)
            # Also update order display
            default_order = {i: ['shallow', 'border', 'avoid', 'collision', 'refugia', 'rheotaxis', 'low_speed', 'wave_drag', 'cohesion', 'alignment'][i] for i in range(10)}
            self.weights_panel.update_order(default_order, current_weights)
        
        # Update plot
        self.control_panel.update_plot(episode + 1, reward)
        
        # Log progress (minimal)
        is_best = "✓ BEST" if reward >= best_reward else ""
        self.control_panel.append_log(f"Ep {episode + 1}/{total}: {reward:.2f} {is_best} | {weight_summary}")
        
        # Update status
        self.control_panel.status_label.setText(f"Episode {episode + 1}/{total} complete")
        
        # Store completed episode for replay (get original data from process_next_queued_episode)
        # We need to retrieve the positions/headings/battery/alive that were used
        # Store in completed_episodes list (with memory optimization)
        if hasattr(self, '_current_episode_data'):
            episode_num, reward, components, positions, headings, battery, alive, weights = self._current_episode_data
            
            # Get storage interval from control panel
            storage_interval = self.control_panel.storage_interval_spin.value()
            
            # Decide whether to store this episode
            should_store = False
            
            # Always store first 5 episodes (show initial chaos)
            if episode_num < 5:
                should_store = True
            # Store every Nth episode
            elif episode_num % storage_interval == 0:
                should_store = True
            
            if should_store:
                self.completed_episodes.append(self._current_episode_data)
                self.control_panel.add_episode_to_list(episode_num)
                print(f"[STORAGE] Stored episode {episode_num} for replay (interval={storage_interval})", flush=True)
            
            # Always track best episode separately
            if self.best_episode_data is None or reward >= self.best_reward:
                self.best_reward = reward
                self.best_episode_data = self._current_episode_data
                print(f"[STORAGE] New best episode: {episode_num} with reward {reward:.2f}", flush=True)
            
            delattr(self, '_current_episode_data')
        else:
            # Safety: if _current_episode_data missing, still try to add to list
            # (shouldn't happen, but prevents episode 0 from being lost)
            self.control_panel.append_log(f"Warning: Episode {episode} missing episode data, adding anyway")
            self.control_panel.add_episode_to_list(episode)
        
        # Clear pending data
        self.pending_episode_data = None
        
        # Mark animation as complete
        self.is_animating = False
        
        # Process next episode from queue if available
        if len(self.episode_queue) > 0:
            self.process_next_queued_episode()
        
    def on_training_completed(self, best_weights: BehavioralWeights, history: List[Tuple[int, float]]):
        """Handle training completion."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("Training completed!")
        self.best_weights = best_weights
        if history:
            self.best_reward = max(r for _, r in history)
            self.control_panel.append_log(f"Best reward: {self.best_reward:.2f}")
        
        if history:
            initial = history[0][1]
            final = history[-1][1]
            improvement = final - initial
            self.control_panel.append_log(f"Initial reward: {initial:.2f}")
            self.control_panel.append_log(f"Final reward: {final:.2f}")
            self.control_panel.append_log(f"Improvement: {improvement:+.2f}")
        
        self.control_panel.status_label.setText("Training complete")
        self.control_panel.set_progress(100, 100)
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
        
    def on_error_occurred(self, error_msg: str):
        """Handle training error."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("ERROR:")
        self.control_panel.append_log(error_msg)
        self.control_panel.status_label.setText("Error occurred")
        
        # Re-enable parameter controls
        self.control_panel.set_parameters_enabled(True)
    
    def closeEvent(self, event):
        """Handle window close - ensure process is stopped properly."""
        if self._is_training_active():
            print("[VIEWER] Stopping training process before close...", flush=True)
            try:
                if self.training_control_queue is not None:
                    self.training_control_queue.put("stop")
                if self.training_process is not None:
                    self.training_process.join(timeout=3.0)
                    if self.training_process.is_alive():
                        print("[VIEWER] WARNING: Process did not stop in time, terminating", flush=True)
                        self.training_process.terminate()
                        self.training_process.join(timeout=1.0)
                self._cleanup_training_process()
            except Exception as e:
                print(f"[VIEWER] Error stopping process: {e}", flush=True)
        
        # Stop animation timer
        if hasattr(self.simulation_canvas, 'replay_widget'):
            try:
                self.simulation_canvas.replay_widget.playing = False
                self.simulation_canvas.replay_widget.timer.stop()
            except Exception:
                pass
        
        print("[VIEWER] Window closed cleanly", flush=True)
        event.accept()
    
    def on_episode_selected(self, index: int):
        """Replay selected episode from dropdown."""
        if index < 0 or index >= len(self.completed_episodes):
            return
        
        episode, reward, components, positions, headings, battery, alive, weights = self.completed_episodes[index]
        
        # Update weights and diagnostics
        total = self.control_panel.episodes_spin.value()
        best_reward = self.best_reward if self.best_reward != float('-inf') else reward
        initial_reward = self.initial_reward if self.initial_reward else reward
        
        self.weights_panel.update_diagnostics(episode + 1, total, reward, best_reward, initial_reward)
        self.weights_panel.update_components(components)
        self.weights_panel.update_weights(weights)
        
        default_order = {i: ['shallow', 'border', 'avoid', 'collision', 'refugia', 'rheotaxis', 'low_speed', 'wave_drag', 'cohesion', 'alignment'][i] for i in range(10)}
        self.weights_panel.update_order(default_order, weights)
        
        # Replay visualization
        self.simulation_canvas.set_positions(positions, headings, battery, alive)
        
        self.control_panel.status_label.setText(f"Replaying episode {episode + 1}")
    
    def on_next_episode(self):
        """Show next episode in list."""
        current_index = self.control_panel.episode_combo.currentIndex()
        if current_index < len(self.completed_episodes) - 1:
            self.control_panel.episode_combo.setCurrentIndex(current_index + 1)
    
    def on_prev_episode(self):
        """Show previous episode in list."""
        current_index = self.control_panel.episode_combo.currentIndex()
        if current_index > 0:
            self.control_panel.episode_combo.setCurrentIndex(current_index - 1)


def main():
    """Run RL training visualizer."""
    print("RL Viewer starting...", flush=True)
    mp.freeze_support()
    try:
        print("Parsing arguments...", flush=True)
        parser = argparse.ArgumentParser(description="RL Training Visualizer for Behavioral Weights")
        parser.add_argument('--model-dir', type=str, required=False, default=None,
                           help='Path to model directory with environment files (depth.tif, vel_x.tif, etc.)')
        parser.add_argument('--start-polygon', type=str, required=False, default=None,
                           help='Path to starting polygon shapefile')
        parser.add_argument('--model-name', type=str, default='salmon_abm',
                           help='Model name (default: salmon_abm)')
        parser.add_argument('--basin', type=str, default='nuyakuk',
                           help='Basin name (default: nuyakuk)')
        
        args = parser.parse_args()
        print(f"Args parsed: model_dir={args.model_dir}, start_polygon={args.start_polygon}", flush=True)
        
        print("Creating QApplication...", flush=True)
        app = QApplication(sys.argv)
        print("Creating RLTrainingViewer...", flush=True)
        viewer = RLTrainingViewer(
            model_dir=args.model_dir,
            start_polygon=args.start_polygon,
            model_name=args.model_name,
            basin=args.basin
        )
        print("Showing viewer...", flush=True)
        viewer.show()
        viewer.raise_()  # Bring to front
        viewer.activateWindow()  # Activate window
        
        # Show startup message if not configured
        if not args.model_dir or not args.start_polygon:
            viewer.control_panel.append_log("=" * 50)
            viewer.control_panel.append_log("Welcome to RL Training Visualizer!")
            viewer.control_panel.append_log("")
            viewer.control_panel.append_log("Please restart with configuration:")
            viewer.control_panel.append_log("  python -m emergent.salmon_abm.rl_training_viewer \\")
            viewer.control_panel.append_log("    --model-dir data/salmon_abm \\")
            viewer.control_panel.append_log("    --start-polygon data/salmon_abm/start_loc_river_right.shp")
            viewer.control_panel.append_log("")
            viewer.control_panel.append_log("Or configure paths before starting training.")
            viewer.control_panel.append_log("=" * 50)
        
        print("Starting event loop...", flush=True)
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        print(f"FATAL ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
