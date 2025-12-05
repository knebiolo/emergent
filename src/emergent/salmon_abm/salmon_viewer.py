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
        # Make left panel resizable by ensuring a sensible minimum and expanding policy
        try:
            left_panel.setMinimumWidth(220)
            left_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        except Exception:
            pass

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
        try:
            right_panel.setMinimumWidth(220)
            right_panel.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        except Exception:
            pass

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

                # Thin nodes in dense HECRAS outputs to avoid overly-dense triangulation
                # Use a grid-based binning approach; user can set `self.sim.tin_thin_resolution` (meters)
                try:
                    base_res = getattr(self.sim, 'tin_thin_resolution', 10.0)
                    if base_res is None or base_res <= 0:
                        base_res = 10.0
                    tin_max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
                    if tin_max_nodes is None or tin_max_nodes <= 0:
                        tin_max_nodes = 5000

                    # Iteratively increase grid cell size until node count <= tin_max_nodes
                    current_res = float(base_res)
                    rep_idx = np.arange(len(wetted_coords))
                    if len(wetted_coords) > 0:
                        while True:
                            gx = (wetted_coords[:, 0] / current_res).astype(int)
                            gy = (wetted_coords[:, 1] / current_res).astype(int)
                            grid = {}
                            for i, (ix, iy) in enumerate(zip(gx, gy)):
                                key = (int(ix), int(iy))
                                grid.setdefault(key, []).append(i)
                            rep = []
                            for key, idxs in grid.items():
                                if len(idxs) == 1:
                                    rep.append(idxs[0])
                                else:
                                    sub = wetted_depth[idxs]
                                    mid = np.argsort(sub)[len(sub) // 2]
                                    rep.append(idxs[mid])
                            rep = np.array(rep, dtype=int)
                            if len(rep) <= tin_max_nodes or current_res > base_res * 64:
                                rep_idx = rep
                                break
                            # increase resolution to thin more aggressively
                            current_res *= 1.5

                        thinned_coords = wetted_coords[rep_idx]
                        thinned_depth = wetted_depth[rep_idx]
                        print(f"Thinned HECRAS nodes: {len(wetted_coords)} -> {len(thinned_coords)} (base_res={base_res}, final_res={current_res})")
                    else:
                        thinned_coords = wetted_coords
                        thinned_depth = wetted_depth
                except Exception:
                    thinned_coords = wetted_coords
                    thinned_depth = wetted_depth

                # Fast IDW rasterization + optional hillshade for quick background
                # Try TIN (triangulated irregular network) rendering first — vector faces colored by depth
                print("Attempting TIN rendering (vector faces) from HECRAS nodes...")
                try:
                    from scipy.spatial import Delaunay
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors
                    # Build Delaunay triangulation on wetted cell centers (thinned)
                    if len(thinned_coords) >= 3:
                        # Cap total nodes for triangulation to avoid excessive work
                        tin_max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
                        if tin_max_nodes is None:
                            tin_max_nodes = 5000
                        # ensure sim attribute exists for user inspection
                        try:
                            setattr(self.sim, 'tin_max_nodes', tin_max_nodes)
                        except Exception:
                            pass
                        if len(thinned_coords) > tin_max_nodes:
                            rng = np.random.default_rng(0)
                            idx_keep = rng.choice(len(thinned_coords), size=tin_max_nodes, replace=False)
                            tri_pts = thinned_coords[idx_keep]
                            tri_vals = thinned_depth[idx_keep]
                            print(f"Capped thinned nodes: {len(thinned_coords)} -> {len(tri_pts)} (max={tin_max_nodes})")
                        else:
                            tri_pts = thinned_coords
                            tri_vals = thinned_depth

                        import time
                        t0 = time.perf_counter()
                        tri = Delaunay(tri_pts)
                        t1 = time.perf_counter()
                        print(f"Delaunay time: {t1-t0:.2f}s for {len(tri_pts)} nodes")
                        tris = tri.simplices
                        norm = mcolors.Normalize(vmin=np.nanpercentile(tri_vals, 1), vmax=np.nanpercentile(tri_vals, 99))
                        cmap = cm.get_cmap('viridis')

                        # Compute triangle filtering parameters
                        try:
                            # compute edge lengths for triangles
                            pts = tri_pts
                            a = pts[tris[:, 0]]
                            b = pts[tris[:, 1]]
                            c = pts[tris[:, 2]]
                            lab = np.linalg.norm(a - b, axis=1)
                            lbc = np.linalg.norm(b - c, axis=1)
                            lca = np.linalg.norm(c - a, axis=1)
                            max_edge = np.maximum(np.maximum(lab, lbc), lca)
                            mean_edge = np.mean(np.concatenate([lab, lbc, lca]))
                        except Exception:
                            max_edge = None
                            mean_edge = None

                        # Default alpha threshold based on mean edge length
                        alpha_param = getattr(self.sim, 'tin_alpha', None)
                        if alpha_param is None or alpha_param <= 0:
                            if mean_edge is not None and mean_edge > 0:
                                alpha_param = mean_edge * 1.5
                            else:
                                alpha_param = getattr(self.sim, 'tin_alpha_fallback', 50.0)

                        kept = np.ones(len(tris), dtype=bool)

                        # Prefer using a centerline-derived buffered perimeter if available
                        kept = np.ones(len(tris), dtype=bool)
                        try:
                            from shapely.geometry import Polygon, Point, MultiPolygon
                            # If a centerline exists on the simulation, build a buffer around it
                            # Prefer vectorized wetted perimeter returned from HECRAS init if present
                            perim_points = None
                            if hasattr(self.sim, '_hecras_geometry_info'):
                                perim_points = self.sim._hecras_geometry_info.get('perimeter_points', None)
                            if perim_points is not None and len(perim_points) > 0:
                                try:
                                    from shapely.geometry import Polygon, Point as ShPoint
                                    perimeter = Polygon([tuple(p) for p in perim_points])
                                except Exception:
                                    perimeter = None
                            elif hasattr(self.sim, 'centerline') and self.sim.centerline is not None:
                                cl = self.sim.centerline
                                # Compute distances from a sample of tri points to centerline to estimate channel half-width
                                sample_pts = tri_pts
                                if len(tri_pts) > 5000:
                                    idxs = np.linspace(0, len(tri_pts)-1, 5000).astype(int)
                                    sample_pts = tri_pts[idxs]
                                from shapely.geometry import Point as ShPoint
                                dists = []
                                for x, y in sample_pts:
                                    try:
                                        dists.append(cl.distance(ShPoint(x, y)))
                                    except Exception:
                                        dists.append(0.0)
                                dists = np.array(dists)
                                if dists.size > 0:
                                    buf_w = max(2.0, np.percentile(dists, 95) * 1.2)
                                else:
                                    buf_w = alpha_param * 1.5
                                perimeter = cl.buffer(buf_w)

                                # Filter triangles by centroid inside perimeter
                                for i, t in enumerate(tris):
                                    if max_edge is not None and max_edge[i] > alpha_param * 4.0:
                                        kept[i] = False
                                        continue
                                    coords = tri_pts[t]
                                    centroid = np.mean(coords, axis=0)
                                    try:
                                        if not perimeter.contains(ShPoint(centroid[0], centroid[1])):
                                            kept[i] = False
                                    except Exception:
                                        if not perimeter.intersects(ShPoint(centroid[0], centroid[1])):
                                            kept[i] = False

                                # Draw perimeter polygon on plot
                                try:
                                    if hasattr(perimeter, 'geoms'):
                                        geoms = perimeter.geoms
                                    else:
                                        geoms = [perimeter]
                                    for g in geoms:
                                        ext = g.exterior.coords[:]
                                        qpoly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in ext])
                                        pen = QtGui.QPen(QtGui.QColor(200, 200, 200), 2)
                                        pen.setStyle(Qt.DotLine)
                                        perimeter_item = QtWidgets.QGraphicsPolygonItem(qpoly)
                                        perimeter_item.setPen(pen)
                                        perimeter_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
                                        self.plot_widget.addItem(perimeter_item)
                                except Exception:
                                    pass
                            else:
                                # shapely available but no centerline; fall back to triangle-union perimeter
                                from shapely.geometry import Polygon, Point as ShPoint, MultiPolygon
                                from shapely.ops import unary_union
                                polygons_to_union = []
                                for i, t in enumerate(tris):
                                    if max_edge is not None and max_edge[i] > alpha_param * 4.0:
                                        kept[i] = False
                                        continue
                                    coords = tri_pts[t]
                                    poly = Polygon([(coords[0,0], coords[0,1]), (coords[1,0], coords[1,1]), (coords[2,0], coords[2,1])])
                                    polygons_to_union.append(poly)
                                if len(polygons_to_union) > 0:
                                    union = unary_union(polygons_to_union)
                                    for i, t in enumerate(tris):
                                        if not kept[i]:
                                            continue
                                        coords = tri_pts[t]
                                        centroid = np.mean(coords, axis=0)
                                        try:
                                            if not union.contains(ShPoint(centroid[0], centroid[1])):
                                                kept[i] = False
                                        except Exception:
                                            if not union.intersects(ShPoint(centroid[0], centroid[1])):
                                                kept[i] = False
                                    try:
                                        if isinstance(union, MultiPolygon):
                                            geoms = union.geoms
                                        else:
                                            geoms = [union]
                                        for g in geoms:
                                            ext = g.exterior.coords[:]
                                            qpoly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in ext])
                                            pen = QtGui.QPen(QtGui.QColor(200, 200, 200), 2)
                                            pen.setStyle(Qt.DotLine)
                                            perimeter_item = QtWidgets.QGraphicsPolygonItem(qpoly)
                                            perimeter_item.setPen(pen)
                                            perimeter_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
                                            self.plot_widget.addItem(perimeter_item)
                                    except Exception:
                                        pass
                        except Exception:
                            # shapely not available; fall back to simple edge-length filtering
                            if max_edge is not None:
                                for i in range(len(tris)):
                                    if max_edge[i] > alpha_param * 3.0:
                                        kept[i] = False

                        # Prepare a coarse IDW raster background to guarantee full coverage (no holes)
                        try:
                            # Build a coarse IDW covering raster from thinned points (fast, low-res)
                            from scipy.spatial import cKDTree
                            nx_bg = 300
                            pts_bg = tri_pts
                            vals_bg = tri_vals
                            minx, miny = pts_bg[:, 0].min(), pts_bg[:, 1].min()
                            maxx, maxy = pts_bg[:, 0].max(), pts_bg[:, 1].max()
                            ny_bg = max(100, int(nx_bg * (maxy - miny) / max(1e-6, (maxx - minx))))
                            xi = np.linspace(minx, maxx, nx_bg)
                            yi = np.linspace(miny, maxy, ny_bg)
                            XI, YI = np.meshgrid(xi, yi)
                            grid_coords = np.column_stack([XI.ravel(), YI.ravel()])
                            tree_bg = cKDTree(pts_bg)
                            k_bg = min(8, len(pts_bg))
                            dists_bg, idxs_bg = tree_bg.query(grid_coords, k=k_bg)
                            if k_bg == 1:
                                dists_bg = dists_bg[:, None]
                                idxs_bg = idxs_bg[:, None]
                            dists_bg[dists_bg == 0] = 1e-6
                            weights_bg = 1.0 / (dists_bg ** 2)
                            neighbor_vals_bg = vals_bg[idxs_bg]
                            weighted_vals_bg = np.sum(weights_bg * neighbor_vals_bg, axis=1) / np.sum(weights_bg, axis=1)
                            Z_bg = weighted_vals_bg.reshape((ny_bg, nx_bg))
                            # simple gaussian smoothing for visual quality
                            try:
                                from scipy.ndimage import gaussian_filter
                                Z_bg = gaussian_filter(Z_bg, sigma=1.0)
                            except Exception:
                                pass
                            # build RGB shaded image
                            import matplotlib.cm as cm
                            import matplotlib.colors as mcolors
                            ls = None
                            try:
                                from matplotlib.colors import LightSource
                                ls = LightSource(azdeg=315, altdeg=45)
                            except Exception:
                                ls = None
                            norm_bg = mcolors.Normalize(vmin=np.nanpercentile(Z_bg, 1), vmax=np.nanpercentile(Z_bg, 99))
                            rgb_bg = cm.get_cmap('viridis')(norm_bg(Z_bg))[:, :, :3]
                            if ls is not None:
                                hill = ls.hillshade(Z_bg, vert_exag=1.0, dx=(maxx - minx)/nx_bg, dy=(maxy - miny)/ny_bg)
                                shaded_bg = rgb_bg * (0.6 + 0.4 * hill[:, :, None])
                            else:
                                shaded_bg = rgb_bg
                            shaded_uint8 = np.clip(shaded_bg * 255, 0, 255).astype('uint8')
                            # alpha fully opaque for background
                            alpha_bg = np.full((ny_bg, nx_bg), 255, dtype='uint8')
                            # transpose to image orientation
                            try:
                                img_arr = np.dstack([np.transpose(shaded_uint8, (1,0,2)), np.transpose(alpha_bg, (1,0))])
                            except Exception:
                                img_arr = np.dstack([shaded_uint8, alpha_bg])
                            bg_item = pg.ImageItem(img_arr)
                            self.plot_widget.addItem(bg_item)
                            bg_item.setZValue(-200)
                            try:
                                bg_item.setRect(minx, miny, (maxx - minx), (maxy - miny))
                            except Exception:
                                try:
                                    bg_item.setPos((minx, miny))
                                    bg_item.scale((maxx - minx) / nx_bg, (maxy - miny) / ny_bg)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Plot filtered triangles on top of the background (ensures visual completeness)
                        kept_tris = tris[kept]
                        for t in kept_tris:
                            coords = tri_pts[t]
                            poly = QtGui.QPolygonF([QtCore.QPointF(coords[0,0], coords[0,1]),
                                                   QtCore.QPointF(coords[1,0], coords[1,1]),
                                                   QtCore.QPointF(coords[2,0], coords[2,1])])
                            mean_depth = np.mean(tri_vals[t])
                            rgba = cmap(norm(mean_depth))
                            color = QtGui.QColor(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 200)
                            item = QtWidgets.QGraphicsPolygonItem(poly)
                            brush = QtGui.QBrush(color)
                            pen = QtGui.QPen(QtCore.Qt.NoPen)
                            item.setBrush(brush)
                            item.setPen(pen)
                            self.plot_widget.addItem(item)
                        print(f"Plotted filtered TIN with {len(kept_tris)} triangles (from {len(tris)})")
                        self.initial_zoom_done = False
                        return
                except Exception:
                    # TIN rendering failed/too sparse — fall back to rasterized IDW below
                    pass

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
                            # Nearest-neighbor fill for NaNs (use full point set)
                            full_tree = cKDTree(pts)
                            d_nn, idx_nn = full_tree.query(grid_coords, k=1)
                            nn_vals = vals[idx_nn].reshape((ny, nx))
                            Z[np.isnan(Z)] = nn_vals[np.isnan(Z)]
                    except Exception:
                        pass

                    # Apply stronger low-pass (gaussian blur) and median filter to reduce banding
                    try:
                        from scipy.ndimage import gaussian_filter, median_filter
                        # stronger smoothing to reduce banding artifacts
                        Z = gaussian_filter(Z, sigma=3.0)
                        Z = median_filter(Z, size=5)
                        # final NN-fill if any NaNs persist
                        if np.isnan(Z).any():
                            full_tree = cKDTree(pts)
                            d_nn, idx_nn = full_tree.query(grid_coords, k=1)
                            nn_vals = vals[idx_nn].reshape((ny, nx))
                            Z[np.isnan(Z)] = nn_vals[np.isnan(Z)]
                    except Exception:
                        try:
                            from scipy.ndimage import gaussian_filter
                            Z = gaussian_filter(Z, sigma=1.5)
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
            except Exception:
                pass
    
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
