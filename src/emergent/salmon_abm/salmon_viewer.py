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


class SalmonViewer(QtWidgets.QWidget):
    """Live visualization for salmon ABM simulations."""
    
    def __init__(self, simulation, dt=0.1, T=600, rl_trainer=None, 
                 show_velocity_field=False, show_depth=True):
        """Initialize salmon viewer.
        
        Parameters
        ----------
        simulation : simulation object
            The salmon ABM simulation instance
        dt : float
            Timestep size in seconds
        T : float
            Total simulation time
        rl_trainer : RLTrainer, optional
            If provided, displays RL training metrics
        show_velocity_field : bool
            Whether to display velocity arrows (expensive for large grids)
        show_depth : bool
            Whether to show depth raster as background
        """
        super().__init__()
        
        self.sim = simulation
        self.dt = dt
        self.T = T
        self.n_timesteps = int(T / dt)
        self.current_timestep = 0
        self.paused = False
        self.rl_trainer = rl_trainer
        self.show_velocity_field = show_velocity_field
        self.show_depth = show_depth
        
        # RL training state
        if self.rl_trainer:
            self.current_episode = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
            self.best_reward = -np.inf
            self.rewards_history = []
        
        # Initialize UI
        self.init_ui()
        
        # Start simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(dt * 1000))  # Convert to milliseconds
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Salmon ABM Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel: plot area
        plot_layout = QVBoxLayout()
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('bottom', 'Easting', units='m')
        self.plot_widget.setLabel('left', 'Northing', units='m')
        plot_layout.addWidget(self.plot_widget)
        
        # Setup background (depth raster)
        if self.show_depth:
            self.setup_background()
        
        # Agent scatter plot
        self.agent_scatter = ScatterPlotItem(
            size=6,
            pen=mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 200),
            symbol='o'
        )
        self.plot_widget.addItem(self.agent_scatter)
        
        # Velocity field arrows (optional)
        if self.show_velocity_field:
            self.setup_velocity_field()
        
        # Status bar
        status_layout = QHBoxLayout()
        self.timestep_label = QLabel(f"Timestep: 0 / {self.n_timesteps}")
        self.time_label = QLabel(f"Time: 0.0 / {self.T:.1f}s")
        self.alive_label = QLabel(f"Alive: {self.sim.num_agents}")
        status_layout.addWidget(self.timestep_label)
        status_layout.addWidget(self.time_label)
        status_layout.addWidget(self.alive_label)
        status_layout.addStretch()
        plot_layout.addLayout(status_layout)
        
        main_layout.addLayout(plot_layout, stretch=3)
        
        # Right panel: controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=1)
        
        self.setLayout(main_layout)
        
    def setup_background(self):
        """Setup depth raster as background image or HECRAS wetted cells."""
        try:
            # Try HECRAS mode first
            if hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path'):
                import h5py
                from scipy.interpolate import griddata
                plan_path = self.sim.hecras_plan_path
                with h5py.File(plan_path, 'r') as hdf:
                    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
                    
                # Mask by wetted perimeter
                wetted_mask = depth > 0.05
                wetted_coords = coords[wetted_mask]
                wetted_depth = depth[wetted_mask]
                
                # Create regular grid for ImageItem
                x_min, x_max = wetted_coords[:, 0].min(), wetted_coords[:, 0].max()
                y_min, y_max = wetted_coords[:, 1].min(), wetted_coords[:, 1].max()
                grid_res = 200  # Resolution
                xi = np.linspace(x_min, x_max, grid_res)
                yi = np.linspace(y_min, y_max, grid_res)
                xi_grid, yi_grid = np.meshgrid(xi, yi)
                
                # Interpolate depth onto grid
                depth_grid = griddata(wetted_coords, wetted_depth, (xi_grid, yi_grid), method='nearest')
                
                # Create ImageItem with viridis colormap
                img = pg.ImageItem(depth_grid, autoRange=False, autoLevels=True)
                img.setRect(QtCore.QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
                img.setZValue(-10)
                colormap = pg.colormap.get('viridis')
                img.setColorMap(colormap)
                self.plot_widget.addItem(img)
                
                # Set view bounds
                self.plot_widget.setXRange(x_min, x_max)
                self.plot_widget.setYRange(y_min, y_max)
                return
            
            # Fallback: load from HDF5
            env = self.sim.hdf5.get('environment')
            if env is None or 'depth' not in env:
                return
                
            depth_arr = env['depth'][:]
            
            # Get transform for georeferencing
            if hasattr(self.sim, 'depth_rast_transform'):
                transform = self.sim.depth_rast_transform
                bounds = rasterio.transform.array_bounds(
                    depth_arr.shape[0], depth_arr.shape[1], transform
                )
                
                # Create ImageItem
                img = pg.ImageItem(depth_arr, autoRange=False, autoLevels=True)
                img.setRect(QtCore.QRectF(bounds[0], bounds[1], 
                                         bounds[2] - bounds[0], 
                                         bounds[3] - bounds[1]))
                img.setZValue(-10)  # Behind agents
                
                # Set colormap
                colormap = pg.colormap.get('viridis')
                img.setColorMap(colormap)
                
                self.plot_widget.addItem(img)
                
                # Set initial view
                self.plot_widget.setXRange(bounds[0], bounds[2])
                self.plot_widget.setYRange(bounds[1], bounds[3])
                
        except Exception as e:
            print(f"Warning: Could not load background: {e}")
    
    def setup_velocity_field(self):
        """Setup velocity field arrows (downsampled for performance)."""
        # TODO: Implement arrow field similar to ship_viewer.ArrowField
        # For now, skip this expensive visualization
        pass
    
    def create_control_panel(self):
        """Create right-side control panel."""
        panel = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_pause)
        layout.addWidget(self.play_pause_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset Simulation")
        reset_btn.clicked.connect(self.reset_simulation)
        layout.addWidget(reset_btn)
        
        # Speed control
        speed_group = QGroupBox("Simulation Speed")
        speed_layout = QVBoxLayout()
        self.speed_label = QLabel("Speed: 1.0x")
        speed_layout.addWidget(self.speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # Agent count display with display options in right column
        agent_group = QGroupBox("Agents")
        agent_layout = QHBoxLayout()
        
        # Left column: counts
        left_col = QVBoxLayout()
        self.agent_count_label = QLabel(f"Total: {self.sim.num_agents}")
        self.alive_count_label = QLabel(f"Alive: {self.sim.num_agents}")
        left_col.addWidget(self.agent_count_label)
        left_col.addWidget(self.alive_count_label)
        
        # Right column: display options
        right_col = QVBoxLayout()
        self.show_trajectories_cb = QCheckBox("Trajectories")
        self.show_trajectories_cb.setChecked(False)
        self.show_dead_cb = QCheckBox("Show Dead")
        self.show_dead_cb.setChecked(True)
        right_col.addWidget(self.show_trajectories_cb)
        right_col.addWidget(self.show_dead_cb)
        
        agent_layout.addLayout(left_col)
        agent_layout.addLayout(right_col)
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
        # Behavioral weights panel
        weights_group = self.create_weights_panel()
        layout.addWidget(weights_group)
        
        # RL Training metrics (if applicable)
        if self.rl_trainer:
            rl_group = self.create_rl_panel()
            layout.addWidget(rl_group)
        
        # Behavior metrics
        metrics_group = self.create_metrics_panel()
        layout.addWidget(metrics_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_rl_panel(self):
        """Create RL training metrics panel."""
        rl_group = QGroupBox("RL Training")
        rl_group.setCheckable(True)
        rl_group.setChecked(True)
        layout = QVBoxLayout()
        
        self.episode_label = QLabel(f"Episode: {self.current_episode} | Timestep: 0")
        self.reward_label = QLabel(f"Reward: {self.episode_reward:.2f}")
        self.best_reward_label = QLabel(f"Best: {self.best_reward:.2f}")
        
        layout.addWidget(self.episode_label)
        layout.addWidget(self.reward_label)
        layout.addWidget(self.best_reward_label)
        
        # Reward plot
        self.reward_plot = pg.PlotWidget(title="Episode Rewards")
        self.reward_plot.setLabel('bottom', 'Episode')
        self.reward_plot.setLabel('left', 'Total Reward')
        self.reward_plot.setMaximumHeight(200)
        layout.addWidget(self.reward_plot)
        
        rl_group.setLayout(layout)
        return rl_group
    
    def create_weights_panel(self):
        """Create behavioral weights display panel with sliders."""
        weights_group = QGroupBox("Behavioral Weights")
        weights_group.setCheckable(True)
        weights_group.setChecked(True)
        layout = QVBoxLayout()
        
        # Get weights from simulation
        if hasattr(self.sim, 'behavioral_weights'):
            weights = self.sim.behavioral_weights
            
            # Create sliders for each weight
            self.weight_labels = {}
            self.weight_sliders = {}
            weight_attrs = ['rheotaxis_weight', 'cohesion_weight', 'separation_weight', 
                          'alignment_weight', 'sog_weight', 'border_cue_weight']
            
            for attr in weight_attrs:
                if hasattr(weights, attr):
                    value = getattr(weights, attr)
                    
                    # Label
                    label = QLabel(f"{attr.replace('_', ' ').title()}: {value:.3f}")
                    label.setStyleSheet("font-size: 9pt;")
                    self.weight_labels[attr] = label
                    layout.addWidget(label)
                    
                    # Slider (0.0 to 2.0, resolution 0.01)
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(0)
                    slider.setMaximum(200)
                    slider.setValue(int(value * 100))
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
        weight_value = value / 100.0
        label.setText(f"{attr.replace('_', ' ').title()}: {weight_value:.3f}")
        if hasattr(self.sim, 'behavioral_weights'):
            setattr(self.sim.behavioral_weights, attr, weight_value)
            self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
    
    def create_metrics_panel(self):
        """Create behavior metrics panel."""
        metrics_group = QGroupBox("Metrics")
        metrics_group.setCheckable(True)
        metrics_group.setChecked(True)
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
        
        # Schooling metrics
        self.mean_nn_dist_label = QLabel("Mean NN Dist: --")
        self.polarization_label = QLabel("Polarization: --")
        layout.addWidget(self.mean_nn_dist_label)
        layout.addWidget(self.polarization_label)
        
        metrics_group.setLayout(layout)
        return metrics_group
    
    def toggle_pause(self):
        """Toggle simulation pause state."""
        self.paused = not self.paused
        self.play_pause_btn.setText("Play" if self.paused else "Pause")
        
    def update_speed(self, value):
        """Update simulation speed."""
        # value ranges 1-100, map to 0.1x - 10x
        speed = value / 10.0
        self.speed_label.setText(f"Speed: {speed:.1f}x")
        
        # Update timer interval
        interval = int(self.dt * 1000 / speed)
        self.timer.setInterval(max(1, interval))
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        if self.rl_trainer:
            self.sim.reset_spatial_state()
        self.current_timestep = 0
        self.update_displays()
    
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
            print(f"Error in simulation update: {e}")
            import traceback
            traceback.print_exc()
            self.paused = True
            self.play_pause_btn.setText("Play")
    
    def update_rl_training(self):
        """Update RL training metrics and episode management."""
        # Extract current state
        current_metrics = self.rl_trainer.extract_state_metrics()
        
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
    
    def update_displays(self):
        """Update all display elements."""
        # Update agent positions
        alive_mask = (self.sim.dead == 0)
        if self.show_dead_cb.isChecked():
            # Show all agents
            x = self.sim.X
            y = self.sim.Y
            # Color by alive/dead
            alive_color = np.array([255, 100, 100, 200])
            dead_color = np.array([100, 100, 100, 100])
            colors = np.where(alive_mask[:, np.newaxis], alive_color, dead_color)
        else:
            # Show only alive
            x = self.sim.X[alive_mask]
            y = self.sim.Y[alive_mask]
            colors = [[255, 100, 100, 200]] * len(x)
        
        self.agent_scatter.setData(x, y, brush=[pg.mkBrush(*c) for c in colors])
        
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
                        # Highlight weights in green during RL mode
                        label.setStyleSheet("font-size: 9pt; color: #00ff00;")
        
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
    
    def run(self):
        """Start the viewer (blocking call)."""
        self.show()
        return QtWidgets.QApplication.instance().exec_()


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
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    return viewer.run()
