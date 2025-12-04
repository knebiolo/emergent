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
        print("Creating SalmonViewer...")
        super().__init__()
        
        print(f"Initializing viewer for {simulation.num_agents} agents...")
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
            print("Setting up RL trainer...")
            self.current_episode = 0
            self.episode_reward = 0.0
            self.prev_metrics = None
            self.best_reward = -np.inf
            self.rewards_history = []
        
        # Initialize UI
        print("About to initialize UI...")
        self.init_ui()
        
        # Start simulation timer (paused initially)
        print("Starting timer (paused)...")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.paused = True  # Start paused
        self.timer.start(int(dt * 1000))  # Convert to milliseconds
        print("SalmonViewer initialization complete!")
        
    def init_ui(self):
        """Initialize the user interface."""
        print("Initializing UI...")
        self.setWindowTitle("Salmon ABM Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel: plot area
        plot_layout = QVBoxLayout()
        
        # Create plot widget
        print("Creating plot widget...")
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('bottom', 'Easting', units='m')
        self.plot_widget.setLabel('left', 'Northing', units='m')
        plot_layout.addWidget(self.plot_widget)
        
        # Setup background (depth raster)
        if self.show_depth:
            self.setup_background()
        
        print("Creating agent scatter plot...")
        # Agent scatter plot
        self.agent_scatter = ScatterPlotItem(
            size=6,
            pen=mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 200),
            symbol='o'
        )
        self.plot_widget.addItem(self.agent_scatter)
        
        # Trajectory lines (initially empty)
        self.trajectory_lines = []
        self.trajectory_history = []  # List of (timestep, positions) tuples
        
        # Velocity field arrows (optional)
        if self.show_velocity_field:
            self.setup_velocity_field()
        
        print("Creating status bar...")
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
        
        # Left panel: RL Training and Metrics
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Middle: Plot window
        main_layout.addLayout(plot_layout, stretch=4)
        
        # Right panel: Controls and Weights
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        self.setLayout(main_layout)
        print("UI initialization complete!")
    
    def create_left_panel(self):
        """Create left control panel with RL Training and Metrics."""
        panel = QGroupBox("Training & Metrics")
        layout = QVBoxLayout()
        
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
    
    def create_right_panel(self):
        """Create right control panel with controls and weights."""
        panel = QGroupBox("Controls & Weights")
        layout = QVBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_pause)
        layout.addWidget(self.play_pause_btn)
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
        
        # Agent count display with display options
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
        self.show_direction_cb = QCheckBox("Direction")
        self.show_direction_cb.setChecked(True)
        right_col.addWidget(self.show_trajectories_cb)
        right_col.addWidget(self.show_dead_cb)
        right_col.addWidget(self.show_direction_cb)
        
        agent_layout.addLayout(left_col)
        agent_layout.addLayout(right_col)
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
        # Behavioral weights panel
        weights_group = self.create_weights_panel()
        layout.addWidget(weights_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def setup_background(self):
        """Setup depth raster as background image or HECRAS wetted cells."""
        try:
            # Try HECRAS mode first
            if hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path'):
                import h5py
                plan_path = self.sim.hecras_plan_path
                with h5py.File(plan_path, 'r') as hdf:
                    coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                    depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
                    face_points = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Face and Point Indexes'][:])
                    points = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'][:])
                    
                # Mask by wetted perimeter
                print("Masking by wetted perimeter...")
                wetted_mask = depth > 0.05
                wetted_coords = coords[wetted_mask]
                wetted_depth = depth[wetted_mask]
                wetted_face_points = face_points[wetted_mask]
                print(f"Wetted cells: {len(wetted_coords)}")
                
                # Calculate cell areas to determine appropriate dot sizes
                print("Calculating cell areas...")
                cell_areas = []
                for fp in wetted_face_points:
                    # Get face point indices
                    face_idx = fp[fp >= 0]  # Filter out -1 padding
                    if len(face_idx) >= 3:
                        cell_points = points[face_idx]
                        # Approximate area using bounding box
                        x_range = cell_points[:, 0].max() - cell_points[:, 0].min()
                        y_range = cell_points[:, 1].max() - cell_points[:, 1].min()
                        area = x_range * y_range
                        cell_areas.append(area)
                    else:
                        cell_areas.append(1.0)  # Fallback
                cell_areas = np.array(cell_areas)
                
                # Plot wetted perimeter with depth coloring (viridis colormap)
                print("Plotting wetted perimeter with depth colors...")
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                
                # Normalize depth for colormap (clip extreme values)
                depth_min, depth_max = np.percentile(wetted_depth, [1, 99])
                norm = mcolors.Normalize(vmin=depth_min, vmax=depth_max)
                cmap = cm.get_cmap('viridis')
                
                # Sample points if too many (plot every Nth point to avoid slowdown)
                sample_step = max(1, len(wetted_coords) // 50000)  # Max ~50k points
                sampled_coords = wetted_coords[::sample_step]
                sampled_depth = wetted_depth[::sample_step]
                sampled_areas = cell_areas[::sample_step]
                
                # Convert depths to colors
                colors = [cmap(norm(d)) for d in sampled_depth]
                colors_rgb = [(int(c[0]*255), int(c[1]*255), int(c[2]*255), 120) for c in colors]
                
                # Calculate adaptive dot sizes based on cell area (scale to sqrt of area)
                # Target: small cells get size ~3, large cells get proportionally larger
                median_area = np.median(sampled_areas)
                dot_sizes = 3.0 * np.sqrt(sampled_areas / median_area)
                dot_sizes = np.clip(dot_sizes, 2, 20)  # Clamp between 2 and 20
                
                # Plot as scatter with adaptive sizes
                scatter = pg.ScatterPlotItem(
                    pos=sampled_coords,
                    size=dot_sizes,
                    brush=[pg.mkBrush(color=c) for c in colors_rgb],
                    pen=None
                )
                self.plot_widget.addItem(scatter)
                print(f"Plotted {len(sampled_coords)} wetted cells with depth colors")
                
                # Plot centerline from simulation
                print("Plotting centerline...")
                if hasattr(self.sim, 'centerline') and self.sim.centerline is not None:
                    from shapely.geometry import LineString
                    if isinstance(self.sim.centerline, LineString):
                        centerline_coords = np.array(self.sim.centerline.coords)
                        self.plot_widget.plot(centerline_coords[:, 0], centerline_coords[:, 1],
                                            pen=pg.mkPen(color=(255, 100, 100), width=3, style=Qt.DashLine))
                        print(f"Plotted centerline with {len(centerline_coords)} points")
                
                print("Background setup complete!")
                # Zoom to agents extent (will be set in update_displays)
                self.initial_zoom_done = False
                return
        except Exception as e:
            print(f"Warning: Could not load background: {e}")
    
    def create_rl_panel(self):
        """Create RL training metrics panel."""
        rl_group = QGroupBox("RL Training")
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
                    label.setStyleSheet("font-size: 9pt; color: black;")
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
        label.setStyleSheet("font-size: 9pt; color: black;")
        if hasattr(self.sim, 'behavioral_weights'):
            setattr(self.sim.behavioral_weights, attr, weight_value)
            self.sim.apply_behavioral_weights(self.sim.behavioral_weights)
    
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
        
        # Schooling metrics
        self.mean_nn_dist_label = QLabel("Mean NN Dist: --")
        self.polarization_label = QLabel("Polarization: --")
        layout.addWidget(self.mean_nn_dist_label)
        layout.addWidget(self.polarization_label)
        
        metrics_group.setLayout(layout)
        return metrics_group
    
    def update_metrics_panel(self, metrics):
        """Update metrics panel labels."""
        self.mean_speed_label.setText(f"Mean Speed: {metrics.get('mean_speed', '--'):.2f}")
        self.max_speed_label.setText(f"Max Speed: {metrics.get('max_speed', '--'):.2f}")
        self.mean_energy_label.setText(f"Mean Energy: {metrics.get('mean_energy', '--'):.2f}")
        self.min_energy_label.setText(f"Min Energy: {metrics.get('min_energy', '--'):.2f}")
        self.upstream_progress_label.setText(f"Upstream Progress: {metrics.get('upstream_progress', '--'):.2f}")
        self.mean_centerline_label.setText(f"Mean Centerline: {metrics.get('mean_centerline', '--'):.2f}")
        self.mean_passage_delay_label.setText(f"Mean Passage Delay: {metrics.get('mean_passage_delay', '--'):.2f}")
        self.mean_nn_dist_label.setText(f"Mean NN Dist: {metrics.get('mean_nn_dist', '--'):.2f}")
        self.polarization_label.setText(f"Polarization: {metrics.get('polarization', '--'):.2f}")
    
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
            
            # Update metrics panel with passage delay
            if hasattr(self.sim, 'get_mean_passage_delay'):
                mean_delay = self.sim.get_mean_passage_delay()
                self.update_metrics_panel({'mean_passage_delay': mean_delay})
                print(f"Mean Passage Delay: {mean_delay:.2f} timesteps")
    
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
        
        # Draw direction indicators (wind sock style)
        if self.show_direction_cb.isChecked() and hasattr(self.sim, 'heading'):
            # Clear old direction lines
            if not hasattr(self, 'direction_lines'):
                self.direction_lines = []
            for line in self.direction_lines:
                self.plot_widget.removeItem(line)
            self.direction_lines = []
            
            # Draw direction line for each visible agent
            if self.show_dead_cb.isChecked():
                headings = self.sim.heading
                pos_x, pos_y = self.sim.X, self.sim.Y
            else:
                headings = self.sim.heading[alive_mask]
                pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]
            
            # Calculate arrow endpoints (5m length trailing behind)
            arrow_length = 5.0
            end_x = pos_x - arrow_length * np.cos(headings)
            end_y = pos_y - arrow_length * np.sin(headings)
            
            # Draw lines from agent position trailing backward
            for i in range(len(pos_x)):
                line_x = [pos_x[i], end_x[i]]
                line_y = [pos_y[i], end_y[i]]
                line = self.plot_widget.plot(line_x, line_y,
                                            pen=pg.mkPen(color=(255, 200, 100, 150), width=1.5))
                self.direction_lines.append(line)
        else:
            # Clear direction lines when disabled
            if hasattr(self, 'direction_lines'):
                for line in self.direction_lines:
                    self.plot_widget.removeItem(line)
                self.direction_lines = []
        
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
            
            # Draw trajectories for each agent
            if len(self.trajectory_history) > 1:
                for agent_idx in range(self.sim.num_agents):
                    # Extract this agent's path
                    agent_x = [pos[1][agent_idx] for pos in self.trajectory_history]
                    agent_y = [pos[2][agent_idx] for pos in self.trajectory_history]
                    
                    # Plot line
                    line = self.plot_widget.plot(agent_x, agent_y,
                                                pen=pg.mkPen(color=(255, 100, 100, 100), width=1))
                    self.trajectory_lines.append(line)
        else:
            # Clear trajectories when disabled
            if hasattr(self, 'trajectory_lines'):
                for line in self.trajectory_lines:
                    self.plot_widget.removeItem(line)
                self.trajectory_lines = []
                self.trajectory_history = []
        
        # Zoom to agents on first update
        if not hasattr(self, 'initial_zoom_done') or not self.initial_zoom_done:
            if len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                # Add 20% padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                padding = 0.2
                self.plot_widget.setXRange(x_min - padding * x_range, x_max + padding * x_range)
                self.plot_widget.setYRange(y_min - padding * y_range, y_max + padding * y_range)
                self.initial_zoom_done = True
        
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
    print("=== INSIDE launch_viewer ===")
    print(f"Creating Qt Application...")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    print(f"Creating SalmonViewer...")
    viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
    print(f"Starting viewer.run()...")
    return viewer.run()
