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
import argparse
from typing import Optional, Dict, List, Tuple
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
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QPainter, QColor, QPen, QFont
except ImportError as e:
    raise ImportError(f"PyQt5 is required for RL training visualizer: {e}")

# Import RL training components
try:
    from emergent.salmon_abm.rl_training import BehavioralWeights, RLTrainer
    from emergent.salmon_abm.simulation import simulation
except ImportError as e:
    raise ImportError(f"Could not import RL training components: {e}")


class SimulationCanvas(QOpenGLWidget):
    """
    Center panel: Real-time simulation visualization.
    
    Displays agent positions during training episodes.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.positions = None  # Shape: (num_agents, 2)
        self.bounds = None  # (min_x, max_x, min_y, max_y)
        self.point_size = 3.0
        self.colors = QColor(50, 150, 255)  # Blue fish
        
    def set_positions(self, positions: np.ndarray, bounds: Optional[Tuple[float, float, float, float]] = None):
        """Update agent positions for next frame."""
        self.positions = positions
        if bounds is not None:
            self.bounds = bounds
        elif positions is not None and len(positions) > 0:
            # Auto-compute bounds with padding
            min_x, min_y = np.min(positions, axis=0)
            max_x, max_y = np.max(positions, axis=0)
            pad_x = (max_x - min_x) * 0.1
            pad_y = (max_y - min_y) * 0.1
            self.bounds = (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)
        self.update()
        
    def paintEvent(self, event):
        """Draw agent positions."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        if self.positions is None or len(self.positions) == 0:
            # Draw "No Data" message
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for simulation data...")
            return
        
        # Get viewport dimensions
        w, h = self.width(), self.height()
        
        # Get bounds
        if self.bounds is None:
            return
        min_x, max_x, min_y, max_y = self.bounds
        
        # Transform positions to screen coordinates
        def to_screen(x, y):
            sx = (x - min_x) / (max_x - min_x) * w
            sy = h - (y - min_y) / (max_y - min_y) * h  # Flip Y
            return sx, sy
        
        # Draw agents
        painter.setPen(QPen(self.colors, self.point_size))
        painter.setBrush(self.colors)
        
        for pos in self.positions:
            sx, sy = to_screen(pos[0], pos[1])
            painter.drawEllipse(int(sx - self.point_size/2), int(sy - self.point_size/2), 
                               int(self.point_size), int(self.point_size))


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
        
        # Weights display (will be populated dynamically)
        self.weights_group = QGroupBox("Current Weights")
        self.weights_layout = QVBoxLayout()
        self.weight_labels = {}
        self.weights_group.setLayout(self.weights_layout)
        layout.addWidget(self.weights_group)
        
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
        """Update displayed weights."""
        # Clear existing labels
        for label in self.weight_labels.values():
            self.weights_layout.removeWidget(label)
            label.deleteLater()
        self.weight_labels.clear()
        
        # Add new weight labels
        weights_dict = weights.to_dict()
        for name, value in sorted(weights_dict.items()):
            if isinstance(value, (int, float)):
                label = QLabel(f"{name}: {value:.1f}")
            else:
                label = QLabel(f"{name}: {value}")
            label.setFont(QFont("Courier New", 9))
            self.weights_layout.addWidget(label)
            self.weight_labels[name] = label
            
    def update_diagnostics(self, episode: int, total_episodes: int, reward: float, 
                          best_reward: float, initial_reward: float):
        """Update training diagnostics."""
        self.episode_label.setText(f"Episode: {episode} / {total_episodes}")
        self.reward_label.setText(f"Current Reward: {reward:.2f}")
        self.best_reward_label.setText(f"Best Reward: {best_reward:.2f}")
        improvement = reward - initial_reward
        self.improvement_label.setText(f"Improvement: {improvement:+.2f}")
        
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


class ControlPanel(QWidget):
    """
    Right panel: Training controls and settings.
    """
    
    # Signals
    start_training = pyqtSignal()
    pause_training = pyqtSignal()
    stop_training = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Training Controls")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Control buttons
        btn_group = QGroupBox("Playback")
        btn_layout = QVBoxLayout()
        
        self.btn_start = QPushButton("▶ Start Training")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_stop = QPushButton("⏹ Stop")
        
        self.btn_start.clicked.connect(self.start_training.emit)
        self.btn_pause.clicked.connect(self.pause_training.emit)
        self.btn_stop.clicked.connect(self.stop_training.emit)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        
        # Episodes
        params_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 1000)
        self.episodes_spin.setValue(50)
        params_layout.addWidget(self.episodes_spin, 0, 1)
        
        # Timesteps per episode
        params_layout.addWidget(QLabel("Timesteps:"), 1, 0)
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(10, 1000)
        self.timesteps_spin.setValue(100)
        params_layout.addWidget(self.timesteps_spin, 1, 1)
        
        # Number of agents
        params_layout.addWidget(QLabel("Agents:"), 2, 0)
        self.agents_spin = QSpinBox()
        self.agents_spin.setRange(10, 1000)
        self.agents_spin.setValue(200)
        params_layout.addWidget(self.agents_spin, 2, 1)
        
        # Exploration noise
        params_layout.addWidget(QLabel("Exploration:"), 3, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 1.0)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.1)
        params_layout.addWidget(self.noise_spin, 3, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
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
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Courier New", 8))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def append_log(self, message: str):
        """Append message to training log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def set_progress(self, current: int, total: int):
        """Update progress bar."""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
        else:
            self.progress_bar.setValue(0)


class TrainingWorker(QObject):
    """
    Background worker for running RL training without blocking UI.
    """
    
    # Signals
    episode_started = pyqtSignal(int)  # episode number
    episode_completed = pyqtSignal(int, float, dict)  # episode, reward, components
    positions_updated = pyqtSignal(object)  # positions array
    training_completed = pyqtSignal(object, list)  # best_weights, history
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, trainer: RLTrainer, num_episodes: int):
        super().__init__()
        self.trainer = trainer
        self.num_episodes = num_episodes
        self.is_paused = False
        self.is_stopped = False
        
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
                
                # Run episode
                positions, headings, velocities, battery, alive = self.trainer.run_episode(current_weights)
                
                # Emit final positions for visualization
                final_positions = positions[-1, :, :]  # Last timestep, all agents
                self.positions_updated.emit(final_positions)
                
                # Compute reward
                from emergent.salmon_abm.rl_training import compute_episode_reward
                reward, components = compute_episode_reward(
                    positions, headings, velocities, alive,
                    body_length=self.trainer.body_length,
                    threat_level=current_weights.threat_level
                )
                
                # Track best
                if reward > self.trainer.best_reward:
                    self.trainer.best_reward = reward
                    self.trainer.best_weights = current_weights
                    
                # Store history
                self.trainer.episode_history.append((episode, float(reward)))
                
                # Emit progress
                self.episode_completed.emit(episode, float(reward), components)
                
                # Mutate for next episode
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
        self.initial_reward = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components."""
        # Create panels
        self.weights_panel = WeightsPanel()
        self.simulation_canvas = SimulationCanvas()
        self.control_panel = ControlPanel()
        
        # Connect control signals
        self.control_panel.start_training.connect(self.on_start_training)
        self.control_panel.pause_training.connect(self.on_pause_training)
        self.control_panel.stop_training.connect(self.on_stop_training)
        
        # Create splitter for three panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.weights_panel)
        splitter.addWidget(self.simulation_canvas)
        splitter.addWidget(self.control_panel)
        
        # Set relative sizes (20% left, 50% center, 30% right)
        splitter.setSizes([280, 700, 420])
        
        self.setCentralWidget(splitter)
        
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
                longitudinal_profile=None,
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
            sim.neighbor_buffer_radius = weights.sensory_range * fish_length_m
            sim.neighbor_buffer_lengths = weights.sensory_range
            
            return sim
        
        return factory_func
        
    def on_start_training(self):
        """Start training with current parameters."""
        if self.training_thread is not None and self.training_thread.isRunning():
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
            # Create simulation factory
            simulation_factory = self.create_simulation_factory(num_agents, num_timesteps)
            
            # Create RL trainer
            initial_weights = BehavioralWeights()
            config = {
                'exploration_noise': exploration_noise,
                'body_length': 0.3,  # 300mm fish
                'dt': 1.0,
                'num_timesteps': num_timesteps
            }
            self.trainer = RLTrainer(
                simulation_factory=simulation_factory,
                initial_weights=initial_weights,
                config=config
            )
            
            self.control_panel.append_log("Trainer initialized successfully")
            
            # Display initial weights
            self.weights_panel.update_weights(initial_weights)
            
            # Create worker thread
            self.training_worker = TrainingWorker(self.trainer, num_episodes)
            self.training_thread = QThread()
            self.training_worker.moveToThread(self.training_thread)
            
            # Connect signals
            self.training_thread.started.connect(self.training_worker.run)
            self.training_worker.episode_started.connect(self.on_episode_started)
            self.training_worker.episode_completed.connect(self.on_episode_completed)
            self.training_worker.positions_updated.connect(self.on_positions_updated)
            self.training_worker.training_completed.connect(self.on_training_completed)
            self.training_worker.error_occurred.connect(self.on_error_occurred)
            self.training_worker.training_completed.connect(self.training_thread.quit)
            self.training_worker.error_occurred.connect(self.training_thread.quit)
            
            # Start training
            self.training_thread.start()
            self.control_panel.append_log("Training thread started")
            
        except Exception as e:
            import traceback
            self.control_panel.append_log(f"ERROR: Failed to start training")
            self.control_panel.append_log(str(e))
            self.control_panel.append_log(traceback.format_exc())
            self.control_panel.status_label.setText("Error")
        
    def on_pause_training(self):
        """Pause/resume training."""
        if self.training_worker is not None:
            if self.training_worker.is_paused:
                self.training_worker.resume()
                self.control_panel.btn_pause.setText("⏸ Pause")
                self.control_panel.append_log("Training resumed")
            else:
                self.training_worker.pause()
                self.control_panel.btn_pause.setText("▶ Resume")
                self.control_panel.append_log("Training paused")
                
    def on_stop_training(self):
        """Stop training."""
        if self.training_worker is not None:
            self.training_worker.stop()
            self.control_panel.append_log("Stopping training...")
            self.control_panel.status_label.setText("Stopping...")
            
    def on_episode_started(self, episode: int):
        """Handle episode start."""
        total = self.control_panel.episodes_spin.value()
        self.control_panel.set_progress(episode, total)
        self.control_panel.status_label.setText(f"Running episode {episode + 1}/{total}")
        
    def on_episode_completed(self, episode: int, reward: float, components: Dict[str, float]):
        """Handle episode completion."""
        total = self.control_panel.episodes_spin.value()
        
        # Update diagnostics
        if self.initial_reward is None:
            self.initial_reward = reward
            
        best_reward = self.trainer.best_reward if self.trainer else reward
        self.weights_panel.update_diagnostics(episode + 1, total, reward, best_reward, self.initial_reward)
        self.weights_panel.update_components(components)
        
        # Update weights display
        if self.trainer and self.trainer.best_weights:
            self.weights_panel.update_weights(self.trainer.best_weights)
        
        # Log progress
        is_best = "✓ BEST" if reward >= best_reward else ""
        self.control_panel.append_log(f"Episode {episode + 1}/{total}: reward={reward:.2f} {is_best}")
        
    def on_positions_updated(self, positions: np.ndarray):
        """Update simulation canvas with new positions."""
        self.simulation_canvas.set_positions(positions)
        
    def on_training_completed(self, best_weights: BehavioralWeights, history: List[Tuple[int, float]]):
        """Handle training completion."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("Training completed!")
        self.control_panel.append_log(f"Best reward: {self.trainer.best_reward:.2f}")
        
        initial = history[0][1]
        final = history[-1][1]
        improvement = final - initial
        self.control_panel.append_log(f"Initial reward: {initial:.2f}")
        self.control_panel.append_log(f"Final reward: {final:.2f}")
        self.control_panel.append_log(f"Improvement: {improvement:+.2f}")
        
        self.control_panel.status_label.setText("Training complete")
        self.control_panel.set_progress(100, 100)
        
    def on_error_occurred(self, error_msg: str):
        """Handle training error."""
        self.control_panel.append_log("=" * 40)
        self.control_panel.append_log("ERROR:")
        self.control_panel.append_log(error_msg)
        self.control_panel.status_label.setText("Error occurred")


def main():
    """Run RL training visualizer."""
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
    
    app = QApplication(sys.argv)
    viewer = RLTrainingViewer(
        model_dir=args.model_dir,
        start_polygon=args.start_polygon,
        model_name=args.model_name,
        basin=args.basin
    )
    viewer.show()
    
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
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
