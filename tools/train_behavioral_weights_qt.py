#!/usr/bin/env python3
"""
RL training with Qt/PyQtGraph visualization (matching ship_abm architecture).

Usage:
    python tools/train_behavioral_weights_qt.py --episodes 10 --timesteps 100 --agents 100
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# Ensure repository root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import simulation, RLTrainer, BehavioralWeights


class RLTrainingViewer(QtWidgets.QMainWindow):
    """Qt-based RL training viewer matching ship_abm architecture."""
    
    def __init__(self, sim, trainer, timesteps, pid, hecras_plan):
        super().__init__()
        self.sim = sim
        self.trainer = trainer
        self.timesteps = timesteps
        self.pid = pid
        self.hecras_plan = hecras_plan
        
        self.current_timestep = 0
        self.paused = True  # Start paused
        self.episode = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.best_reward = -np.inf
        
        self.setWindowTitle('RL Behavioral Weight Training')
        self.resize(1600, 900)
        
        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton('Start')
        self.pause_btn = QtWidgets.QPushButton('Pause')
        self.reset_btn = QtWidgets.QPushButton('Reset')
        
        self.start_btn.clicked.connect(self.on_start)
        self.pause_btn.clicked.connect(self.on_pause)
        self.reset_btn.clicked.connect(self.on_reset)
        
        # Style buttons
        self.start_btn.setStyleSheet('background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;')
        self.pause_btn.setStyleSheet('background-color: #FF9800; color: white; font-weight: bold; padding: 8px;')
        self.reset_btn.setStyleSheet('background-color: #f44336; color: white; font-weight: bold; padding: 8px;')
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        
        # Status label
        self.status_label = QtWidgets.QLabel('Ready - Press Start')
        self.status_label.setStyleSheet('font-size: 14px; padding: 5px;')
        button_layout.addWidget(self.status_label)
        
        layout.addLayout(button_layout)
        
        # Graphics view
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)
        
        # Agent scatter plot
        self.agent_scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 255, 200))
        self.plot_widget.addItem(self.agent_scatter)
        
        # Agent direction arrows
        self.arrow_lines = []
        
        # Set view to agent extent
        import h5py
        with h5py.File(hecras_plan, 'r') as hdf:
            pts = np.array(hdf.get('Geometry/2D Flow Areas/2D area/Cells Center Coordinate'))
            xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        
        self.plot_widget.setXRange(xmin, xmax)
        self.plot_widget.setYRange(ymin, ymax)
        
        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(50)  # 20 FPS
        
        self.update_agents()
        
    def on_start(self):
        self.paused = False
        self.timer.start()
        self.status_label.setText(f'Running - Episode {self.episode} | t={self.current_timestep}/{self.timesteps}')
        print('Started/Resumed')
        
    def on_pause(self):
        self.paused = True
        self.timer.stop()
        self.status_label.setText(f'Paused - Episode {self.episode} | t={self.current_timestep}/{self.timesteps}')
        print('Paused')
        
    def on_reset(self):
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        self.update_agents()
        self.status_label.setText(f'Reset - Episode {self.episode} | t=0/{self.timesteps}')
        print('Reset to timestep 0')
        
    def update_agents(self):
        """Update agent positions and directions."""
        if not hasattr(self.sim, 'X') or not hasattr(self.sim, 'Y'):
            return
            
        positions = np.column_stack([self.sim.X.flatten(), self.sim.Y.flatten()])
        headings = self.sim.heading.flatten()
        
        # Filter valid agents
        mask = np.isfinite(positions).all(axis=1) & np.isfinite(headings)
        if hasattr(self.sim, 'dead'):
            mask &= (self.sim.dead.flatten() == 0)
        
        positions = positions[mask]
        headings = headings[mask]
        
        if len(positions) == 0:
            return
            
        # Update scatter plot
        self.agent_scatter.setData(positions[:, 0], positions[:, 1])
        
        # Update direction arrows
        for item in self.arrow_lines:
            self.plot_widget.removeItem(item)
        self.arrow_lines = []
        
        shaft_length = 3.0  # meters
        for i in range(len(positions)):
            x, y = positions[i]
            h = headings[i]
            
            # Shaft: line behind agent
            dx_shaft = -shaft_length * np.cos(h)
            dy_shaft = -shaft_length * np.sin(h)
            
            arrow = pg.ArrowItem(angle=np.degrees(h), tipAngle=30, headLen=20, 
                                tailLen=shaft_length*10, tailWidth=2,
                                pen=pg.mkPen(color=(0, 200, 200), width=2),
                                brush=pg.mkBrush(0, 255, 255))
            arrow.setPos(x, y)
            self.plot_widget.addItem(arrow)
            self.arrow_lines.append(arrow)
        
    def update_simulation(self):
        """Run one simulation timestep."""
        if self.paused or self.current_timestep >= self.timesteps:
            return
            
        try:
            # Run timestep
            self.sim.timestep(self.current_timestep, 1.0, 9.81, self.pid)
            
            # RL metrics
            current_metrics = self.trainer.extract_state_metrics()
            if self.prev_metrics is not None:
                reward = self.trainer.compute_reward(self.prev_metrics, current_metrics)
                self.episode_reward += reward
            self.prev_metrics = current_metrics
            
            self.current_timestep += 1
            
            # Update display
            self.update_agents()
            self.status_label.setText(f'Running - Episode {self.episode} | t={self.current_timestep}/{self.timesteps} | Reward: {self.episode_reward:.1f}')
            
            # Episode complete
            if self.current_timestep >= self.timesteps:
                self.on_episode_complete()
                
        except Exception as e:
            print(f'Simulation error at t={self.current_timestep}: {e}')
            import traceback
            traceback.print_exc()
            self.on_pause()
            
    def on_episode_complete(self):
        """Handle episode completion."""
        print(f'\\n{"="*80}')
        print(f'EPISODE {self.episode} COMPLETE | Total Reward: {self.episode_reward:.2f}')
        
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            print(f'🏆 NEW BEST REWARD: {self.best_reward:.2f}')
            
            # Save best weights
            save_path = os.path.join(REPO_ROOT, 'outputs', 'rl_training', 'best_weights.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.trainer.behavioral_weights.to_dict(), f, indent=2)
            print(f'Saved best weights to {save_path}')
        
        print(f'{"="*80}\\n')
        
        # Prepare next episode
        self.episode += 1
        self.current_timestep = 0
        self.episode_reward = 0.0
        self.prev_metrics = None
        
        # Mutate weights
        self.trainer.mutate_behavioral_weights()
        self.trainer.apply_weights_to_simulation()
        
        # Reset simulation spatial state
        try:
            self.sim.reset_spatial_state()
            print(f'Mutated weights for episode {self.episode}: cohesion={self.trainer.behavioral_weights.cohesion:.2f}, separation={self.trainer.behavioral_weights.separation:.2f}')
        except Exception as e:
            print(f'Warning: Could not reset spatial state: {e}')
        
        self.status_label.setText(f'Episode {self.episode - 1} complete! Reward: {self.episode_reward:.1f} - Press Start for Episode {self.episode}')
        self.on_pause()


def setup_training_simulation(args):
    """Initialize simulation and RL trainer."""
    
    # HECRAS plan path
    hecras_plan = os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Production_.p05.hdf')
    
    if not os.path.exists(hecras_plan):
        raise FileNotFoundError(f"HECRAS plan not found: {hecras_plan}")
    
    print(f'Using HECRAS plan: {hecras_plan}')
    print(f'Initializing training simulation with {args.agents} agents...')
    
    # Simulation configuration
    config = {
        'model_dir': os.path.join(REPO_ROOT, 'outputs', 'rl_training'),
        'model_name': 'rl_training_qt',
        'crs': 'EPSG:32605',
        'basin': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Bathymetry.shp'),
        'water_temp': 10.0,
        'start_polygon': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'Nuyakuk_Start_Polygon.shp'),
        'longitudinal_profile': os.path.join(REPO_ROOT, 'data', 'salmon_abm', '20240506', 'nuyakuk_centerline.shp'),
        'env_files': None,  # Using HECRAS instead
        'num_agents': args.agents,
        'fish_length': args.fish_length,
        'hecras_plan_path': hecras_plan,
        'use_hecras': True,
        'hecras_k': 1,
    }
    
    # Create output directory
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Create simulation
    sim = simulation(**config)
    print('Simulation initialized.')
    
    # Create RL trainer
    print('Setting up RL trainer...')
    trainer = RLTrainer(sim)
    trainer.apply_weights_to_simulation()
    print(f'Applied behavioral weights: cohesion={sim.cohesion_weight:.2f}, separation={sim.separation_weight:.2f}, border={sim.border_weight}')
    
    # Setup PID controller
    from src.emergent.salmon_abm.sockeye_SoA_OpenGL_RL import PID_controller
    pid = PID_controller(sim.num_agents, k_p=1.0, k_i=0.0, k_d=0.0)
    
    return sim, trainer, hecras_plan, pid


def main():
    parser = argparse.ArgumentParser(description='Train behavioral weights with Qt visualization')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps per episode')
    parser.add_argument('--agents', type=int, default=100, help='Number of agents')
    parser.add_argument('--fish-length', type=int, default=450, help='Fish length (mm)')
    
    args = parser.parse_args()
    
    print('='*80)
    print('RL BEHAVIORAL WEIGHT TRAINING (Qt Mode)')
    print('='*80)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Timesteps per episode: {args.timesteps}')
    print(f'  Agents: {args.agents}')
    print(f'  Fish length: {args.fish_length} mm')
    print('='*80)
    print()
    
    # Setup simulation
    sim, trainer, hecras_plan, pid = setup_training_simulation(args)
    
    # Launch Qt application
    app = QtWidgets.QApplication(sys.argv)
    viewer = RLTrainingViewer(sim, trainer, args.timesteps, pid, hecras_plan)
    viewer.show()
    
    print('\\nQt viewer launched!')
    print('Click the green "Start" button to begin training\\n')
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
