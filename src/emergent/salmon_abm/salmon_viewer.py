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
        
        # Main layout using QSplitter for resizable panels
        main_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Left panel area (RL + Metrics)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Create central plot widget container
        plot_container = QtWidgets.QWidget()
        plot_layout = QVBoxLayout(plot_container)
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
        
        # Add plot container to splitter
        main_splitter.addWidget(plot_container)

        # Right panel: Controls and Weights
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set initial sizes (left, center, right)
        main_splitter.setSizes([300, 900, 300])

        # Set splitter as the main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
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
        
        # Play / Pause / Reset buttons in a single row
        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        btn_row.addWidget(self.pause_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_simulation)
        btn_row.addWidget(reset_btn)

        layout.addLayout(btn_row)
        
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
                    # Use documented HECRAS dataset names
                    face_info = None
                    points = None
                    if 'Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info' in hdf:
                        face_info = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Face and Orientation Info'][:])
                    if 'Geometry/2D Flow Areas/2D area/FacePoints Coordinate' in hdf:
                        points = np.array(hdf['Geometry/2D Flow Areas/2D area/FacePoints Coordinate'][:])
                    
                # Mask by wetted perimeter
                print("Masking by wetted perimeter...")
                wetted_mask = depth > 0.05
                wetted_coords = coords[wetted_mask]
                wetted_depth = depth[wetted_mask]
                # If face_info/points present, compute areas; otherwise skip adaptive sizing
                wetted_face_points = None
                if face_info is not None and points is not None:
                    wetted_face_points = face_info[wetted_mask]
                print(f"Wetted cells: {len(wetted_coords)}")
                
                # Calculate cell areas to determine appropriate dot sizes when geometry is available
                cell_areas = None
                if wetted_face_points is not None:
                    print("Calculating cell areas...")
                    cell_areas = []
                    for fp in wetted_face_points:
                        face_idx = fp[fp >= 0]  # Filter out -1 padding
                        if len(face_idx) >= 3:
                            cell_points = points[face_idx]
                            # Approximate area using bounding box
                            x_range = cell_points[:, 0].max() - cell_points[:, 0].min()
                            y_range = cell_points[:, 1].max() - cell_points[:, 1].min()
                            area = x_range * y_range
                            cell_areas.append(area)
                        else:
                            cell_areas.append(1.0)
                    cell_areas = np.array(cell_areas)
                
                # Fast IDW rasterization + optional hillshade for quick background
                print("Rasterizing wetted depth via fast IDW and applying hillshade...")
                try:
                    from scipy.spatial import cKDTree
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors
                    from matplotlib.colors import LightSource

                    pts = wetted_coords
                    vals = wetted_depth

                    # Sample points to reduce compute if necessary
                    max_pts = 50000
                    if len(pts) > max_pts:
                        rng = np.random.default_rng(0)
                        idx_sample = rng.choice(len(pts), size=max_pts, replace=False)
                        pts_s = pts[idx_sample]
                        vals_s = vals[idx_sample]
                    else:
                        pts_s = pts
                        vals_s = vals

                    # Create grid size based on extent but limited resolution for speed
                    minx, miny = pts_s[:, 0].min(), pts_s[:, 1].min()
                    maxx, maxy = pts_s[:, 0].max(), pts_s[:, 1].max()
                    nx = 600
                    ny = max(200, int(nx * (maxy - miny) / (maxx - minx)))
                    xi = np.linspace(minx, maxx, nx)
                    yi = np.linspace(miny, maxy, ny)
                    XI, YI = np.meshgrid(xi, yi)
                    grid_coords = np.column_stack([XI.ravel(), YI.ravel()])

                    # Fast IDW: k-NN weighting
                    tree = cKDTree(pts_s)
                    k = min(8, len(pts_s))
                    dists, idxs = tree.query(grid_coords, k=k)
                    # Ensure shapes are 2D when k==1
                    if k == 1:
                        dists = dists[:, np.newaxis]
                        idxs = idxs[:, np.newaxis]
                    # Avoid zero distances
                    dists[dists == 0] = 1e-6
                    weights = 1.0 / (dists ** 2)
                    # vals_s indexed by idxs -> shape (npoints, k)
                    neighbor_vals = vals_s[idxs]
                    weighted_vals = np.sum(weights * neighbor_vals, axis=1) / np.sum(weights, axis=1)
                    Z = weighted_vals.reshape((ny, nx))

                    # Fill holes (cells with NaN or very large distance) using nearest neighbor fallback
                    try:
                        if np.isnan(Z).any():
                            full_tree = cKDTree(pts_s)
                            d_nn, idx_nn = full_tree.query(grid_coords, k=1)
                            nn_vals = vals_s[idx_nn].reshape((ny, nx))
                            Z[np.isnan(Z)] = nn_vals[np.isnan(Z)]
                    except Exception:
                        pass

                    # Apply light low-pass (gaussian blur via convolution) to remove speckle
                    try:
                        from scipy.ndimage import gaussian_filter
                        Z = gaussian_filter(Z, sigma=1.0)
                    except Exception:
                        pass

                    # Apply hillshade once at initialization
                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = cm.get_cmap('viridis')(mcolors.Normalize(vmin=np.nanpercentile(Z, 1), vmax=np.nanpercentile(Z, 99))(Z))
                    hill = ls.hillshade(Z, vert_exag=1.0, dx=(maxx - minx)/nx, dy=(maxy - miny)/ny)
                    shaded = rgb[:, :, :3] * (0.6 + 0.4 * hill[:, :, None])

                    # Prepare RGBA image
                    alpha_channel = np.full((ny, nx), 255, dtype='uint8')

                    # Build a wetted-area mask on the grid by measuring distance to nearest wetted cell center
                    try:
                        full_tree = cKDTree(pts)
                        d_full, _ = full_tree.query(grid_coords, k=1)
                        spacing = np.sqrt(((maxx - minx) * (maxy - miny)) / max(1, len(pts)))
                        wetted_grid_mask = (d_full.reshape((ny, nx)) <= (spacing * 1.5))
                        alpha_channel = (wetted_grid_mask * 255).astype('uint8')
                    except Exception:
                        # If building a full tree fails, keep alpha fully opaque
                        pass

                    shaded_uint8 = np.clip(shaded * 255, 0, 255).astype('uint8')
                    # transpose arrays so rows/cols match plotting coordinate orientation
                    try:
                        # transpose X/Y axes while preserving color channel order
                        shaded_uint8 = np.transpose(shaded_uint8, (1, 0, 2))
                        alpha_channel = np.transpose(alpha_channel, (1, 0))
                    except Exception:
                        try:
                            shaded_uint8 = shaded_uint8.T
                            alpha_channel = alpha_channel.T
                        except Exception:
                            pass
                    rgba_uint8 = np.dstack([shaded_uint8, alpha_channel])

                    img_item = pg.ImageItem(rgba_uint8)
                    self.plot_widget.addItem(img_item)
                    img_item.setZValue(-100)
                    img_item.setOpacity(1.0)
                    # position the image using setRect(minx, miny, width, height)
                    width = (maxx - minx)
                    height = (maxy - miny)
                    try:
                        img_item.setRect(minx, miny, width, height)
                    except TypeError:
                        try:
                            # Some pyqtgraph versions expect QRectF
                            rect = QtCore.QRectF(minx, miny, width, height)
                            img_item.setRect(rect)
                        except Exception:
                            # Fallback to legacy setPos/scale
                            img_item.setPos((minx, miny))
                            try:
                                img_item.scale(width / nx, height / ny)
                            except Exception:
                                pass

                    print(f"Rendered IDW raster ({nx}x{ny})")
                except Exception as e:
                    print(f"IDW rasterization failed: {e}")

                # Plot centerline from simulation (overlay)
                print("Plotting centerline...")
                if hasattr(self.sim, 'centerline') and self.sim.centerline is not None:
                    from shapely.geometry import LineString
                    if isinstance(self.sim.centerline, LineString):
                        centerline_coords = np.array(self.sim.centerline.coords)
                        self.plot_widget.plot(centerline_coords[:, 0], centerline_coords[:, 1],
                                            pen=pg.mkPen(color=(255, 100, 100), width=3, style=Qt.DashLine))
                        print(f"Plotted centerline with {len(centerline_coords)} points")

                self.initial_zoom_done = False
                return
            # Fallback: try ABM rasterized depth if present
            if hasattr(self.sim, 'hdf') and 'environment/depth' in self.sim.hdf:
                try:
                    depth_raster = np.array(self.sim.hdf['environment/depth'][:])
                    x_coords = np.array(self.sim.hdf['x_coords'][:])
                    y_coords = np.array(self.sim.hdf['y_coords'][:])
                    img = pg.ImageItem(depth_raster)
                    img.setOpacity(0.7)
                    self.plot_widget.addItem(img)
                    print("Plotted ABM raster depth fallback")
                    self.initial_zoom_done = False
                    return
                except Exception as e:
                    print(f"Fallback raster plotting failed: {e}")
                
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

        # Collision time-series plot
        self.collision_plot = pg.PlotWidget(title="Collisions Over Time")
        self.collision_plot.setLabel('bottom', 'Timestep')
        self.collision_plot.setLabel('left', 'Collisions')
        self.collision_plot.setMaximumHeight(140)
        self.collision_curve = self.collision_plot.plot([], [], pen=mkPen('r', width=2))
        self.collision_history = []
        layout.addWidget(self.collision_plot)
        
        metrics_group.setLayout(layout)
        return metrics_group
    
    def update_metrics_panel(self, metrics):
        """Update metrics panel labels."""
        self.mean_speed_label.setText(f"Mean Speed: {metrics.get('mean_speed', '--'):.2f}")
        self.max_speed_label.setText(f"Max Speed: {metrics.get('max_speed', '--'):.2f}")
        self.mean_energy_label.setText(f"Mean Energy: {metrics.get('mean_energy', '--'):.2f}")
        self.min_energy_label.setText(f"Min Energy: {metrics.get('min_energy', '--'):.2f}")
        # Upstream progress may be reported as 'upstream_progress' or 'mean_upstream_progress'
        upstream_val = metrics.get('upstream_progress', metrics.get('mean_upstream_progress', '--'))
        if isinstance(upstream_val, (int, float)):
            self.upstream_progress_label.setText(f"Upstream Progress: {upstream_val:.2f}")
        else:
            self.upstream_progress_label.setText(f"Upstream Progress: --")
        self.mean_centerline_label.setText(f"Mean Centerline: {metrics.get('mean_centerline', '--'):.2f}")
        self.mean_passage_delay_label.setText(f"Mean Passage Delay: {metrics.get('mean_passage_delay', '--'):.2f}")
        # Optional passage stats
        if 'passage_success_rate' in metrics:
            self.passage_success_rate_label.setText(f"Passage Success: {metrics['passage_success_rate']:.1%}")
        elif 'success_rate' in metrics:
            self.passage_success_rate_label.setText(f"Passage Success: {metrics['success_rate']:.1%}")
        self.mean_nn_dist_label.setText(f"Mean NN Dist: {metrics.get('mean_nn_dist', '--'):.2f}")
        self.polarization_label.setText(f"Polarization: {metrics.get('polarization', '--'):.2f}")
        # Update collision time-series if provided
        try:
            if 'collision_count' in metrics:
                self.collision_history.append(metrics['collision_count'])
                x = list(range(len(self.collision_history)))
                y = self.collision_history
                self.collision_curve.setData(x, y)
                # keep history length reasonable
                if len(self.collision_history) > 500:
                    self.collision_history.pop(0)
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
    
    def toggle_pause(self):
        """Toggle simulation pause state."""
        self.paused = not self.paused
        # Update Play/Pause button labels
        try:
            self.play_btn.setText("Play" if self.paused else "Pause")
            self.pause_btn.setText("Pause" if self.paused else "Play")
        except Exception:
            pass
        # Ensure play/pause also toggles the timer
        if hasattr(self, 'timer'):
            if self.paused:
                self.timer.stop()
            else:
                self.timer.start(int(self.dt * 1000))

    def reset_simulation(self):
        """Reset simulation to initial state for new episode."""
        if hasattr(self.sim, 'reset_spatial_state'):
            self.sim.reset_spatial_state(reset_positions=True)
        self.current_timestep = 0
        self.current_episode = 1
        self.episode_reward = 0.0
        # refresh display
        self.update_displays()
        
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
            try:
                self.play_btn.setText("Play")
                self.pause_btn.setText("Pause")
            except Exception:
                pass
    
    def update_rl_training(self):
        """Update RL training metrics and episode management."""
        # Extract current state
        current_metrics = self.rl_trainer.extract_state_metrics()
        # Update metrics panel immediately so time-series (collisions) are plotted
        try:
            self.update_metrics_panel(current_metrics)
        except Exception:
            pass
        
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

        # Refresh RL labels and metrics display
        try:
            self.refresh_rl_labels()
        except Exception:
            pass
            
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
