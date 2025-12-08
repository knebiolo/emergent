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
        try:
            # Build vertices (x, y, z)
            z = (np.nan_to_num(self.tri_vals, nan=0.0) * self.vert_exag).astype(float)
            verts = np.column_stack([self.tri_pts[:, 0], self.tri_pts[:, 1], z]).astype(np.float32)
            faces = np.array(self.kept_tris, dtype=np.int32)

            # Map values to colors using a matplotlib colormap if available
            try:
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                vmin = np.nanmin(self.tri_vals)
                vmax = np.nanmax(self.tri_vals)
                vrange = vmax - vmin if (vmax - vmin) != 0 else 1.0
                norm_vals = (self.tri_vals - vmin) / vrange
                cmap = cm.get_cmap('viridis')
                colors = (cmap(norm_vals) * 255).astype(np.uint8)
            except Exception:
                # fallback to gray
                colors = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (len(verts), 1))

            self.mesh_ready.emit({'verts': verts, 'faces': faces, 'colors': colors})
        except Exception as e:
            print('GLMeshBuilder error:', e)


# SalmonViewer widget definition
class SalmonViewer(QtWidgets.QWidget):
    """Main viewer widget for simulation visualization."""

    def __init__(self, simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
        super().__init__()
        self.sim = simulation
        self.dt = dt
        self.T = T
        self.rl_trainer = rl_trainer
        self.weight_sliders = {}
        self.plot_widget = None
        self.setWindowTitle('Salmon ABM Viewer')

        # Build UI
        self.init_ui()

    def init_ui(self):
        """Create main UI: left panel, central plot widget, right panel."""
        main_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Left panel
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Center plot container
        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)
        plot_container.setLayout(plot_layout)
        main_splitter.addWidget(plot_container)

        # Right panel
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
        self.show_tail_cb = QCheckBox("Tail")
        self.show_tail_cb.setChecked(True)
        right_col.addWidget(self.show_trajectories_cb)
        right_col.addWidget(self.show_dead_cb)
        right_col.addWidget(self.show_direction_cb)
        right_col.addWidget(self.show_tail_cb)
        
        agent_layout.addLayout(left_col)
        agent_layout.addLayout(right_col)
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
        # Behavioral weights panel
        weights_group = self.create_weights_panel()
        layout.addWidget(weights_group)

        # Vertical exaggeration control for GL mesh
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
                        # cache for rebuild actions
                        self.last_thinned_coords = thinned_coords
                        self.last_thinned_vals = thinned_depth
                        print(f"Thinned HECRAS nodes: {len(wetted_coords)} -> {len(thinned_coords)} (base_res={base_res}, final_res={current_res})")
                    else:
                        thinned_coords = wetted_coords
                        thinned_depth = wetted_depth
                        self.last_thinned_coords = thinned_coords
                        self.last_thinned_vals = thinned_depth
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
                        # OpenGL-only path: build mesh in background thread and swap into GL view
                        if gl is None:
                            raise RuntimeError('pyqtgraph.opengl (PyOpenGL) not available')

                        # Create GL view if needed and replace the PlotWidget
                        if not hasattr(self, 'gl_view'):
                            self.gl_view = gl.GLViewWidget()
                            # adjust viewpoint and axes
                            try:
                                self.gl_view.opts['distance'] = max(100.0, float(np.max(tri_pts[:,0]) - np.min(tri_pts[:,0]), ))
                            except Exception:
                                self.gl_view.opts['distance'] = 100.0
                            parent = self.plot_widget.parent()
                            layout = parent.layout()
                            layout.removeWidget(self.plot_widget)
                            self.plot_widget.hide()
                            layout.addWidget(self.gl_view)

                        # Launch background mesh builder to avoid blocking UI
                        builder = _GLMeshBuilder(tri_pts, tri_vals, kept_tris, parent=self)

                        def _on_mesh_ready(payload):
                            if 'error' in payload:
                                print('GL mesh builder error:', payload['error'])
                                return
                            verts = payload['verts']
                            faces = payload['faces']
                            colors = payload['colors']
                            try:
                                meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                                mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque')
                                mesh.setGLOptions('opaque')
                                # remove previous mesh if present
                                try:
                                    if hasattr(self, 'tin_mesh') and self.tin_mesh is not None:
                                        self.gl_view.removeItem(self.tin_mesh)
                                except Exception:
                                    pass
                                self.tin_mesh = mesh
                                self.gl_view.addItem(mesh)
                                print(f"Plotted GLMesh TIN with {len(faces)} triangles (from {len(tris)})")
                            except Exception as e:
                                print('Failed to add GL mesh to scene:', e)

                        builder.mesh_ready.connect(_on_mesh_ready)
                        builder.start()
                        self.initial_zoom_done = False
                        return
                except Exception as e:
                    # TIN rendering failed or was too sparse — log and fall back to raster/centerline
                    print(f"TIN rendering failed: {e} -- falling back to raster/centerline display if available")
                    # attempt to plot centerline if available, otherwise continue to raster fallback below
                    try:
                        if hasattr(self.sim, 'centerline') and self.sim.centerline is not None:
                            from shapely.geometry import LineString
                            if isinstance(self.sim.centerline, LineString):
                                centerline_coords = np.array(self.sim.centerline.coords)
                                self.plot_widget.plot(centerline_coords[:, 0], centerline_coords[:, 1],
                                                      pen=pg.mkPen(color=(255, 100, 100), width=3, style=Qt.DashLine))
                                print(f"Plotted centerline with {len(centerline_coords)} points (after TIN failure)")
                    except Exception:
                        pass

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

    def render_tin_from_arrays(self, coords, vals, cap_nodes=5000, vert_exag=None):
        """Render a TIN from provided 2D coords and scalar values using the OpenGL path.

        This helper bypasses the HECRAS file parsing and runs the Delaunay->GL pipeline
        used by `setup_background()`. It is intended for programmatic tests and
        interactive use where arrays are already available.
        """
        try:
            if len(coords) < 3:
                print('Not enough points for TIN')
                return
            tri_pts = np.asarray(coords, dtype=float)
            tri_vals = np.asarray(vals, dtype=float)
            if len(tri_pts) > cap_nodes:
                rng = np.random.default_rng(0)
                idx_keep = rng.choice(len(tri_pts), size=cap_nodes, replace=False)
                tri_pts = tri_pts[idx_keep]
                tri_vals = tri_vals[idx_keep]
                print(f'Capped input nodes: {len(coords)} -> {len(tri_pts)} (max={cap_nodes})')

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
                if perim_points is not None and len(perim_points) > 3:
                    from shapely.geometry import Point as ShPoint, Polygon
                    perimeter = Polygon([tuple(p) for p in perim_points])
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
                self.gl_view = gl.GLViewWidget()
                parent = self.plot_widget.parent()
                layout = parent.layout()
                layout.removeWidget(self.plot_widget)
                self.plot_widget.hide()
                layout.addWidget(self.gl_view)

            # compute vertical exaggeration
            if vert_exag is None:
                vert_exag = getattr(self.sim, 'vert_exag', 1.0)

            # launch mesh builder with vertical exaggeration
            builder = _GLMeshBuilder(tri_pts, tri_vals, kept_tris, vert_exag=vert_exag, parent=self)

            def _on_mesh_ready(payload):
                if 'error' in payload:
                    print('GL mesh builder error:', payload['error'])
                    return
                verts = payload['verts']
                faces = payload['faces']
                colors = payload['colors']
                try:
                    meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                    mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque')
                    mesh.setGLOptions('opaque')
                    try:
                        if hasattr(self, 'tin_mesh') and self.tin_mesh is not None:
                            self.gl_view.removeItem(self.tin_mesh)
                    except Exception:
                        pass
                    self.tin_mesh = mesh
                    self.gl_view.addItem(mesh)
                    print(f'Plotted GLMesh TIN with {len(faces)} triangles')
                except Exception as e:
                    print('Failed to add GL mesh to scene:', e)

            builder.mesh_ready.connect(_on_mesh_ready)
            builder.start()

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
        # tighter packing to allow more sliders to fit vertically
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Get weights from simulation
        if hasattr(self.sim, 'behavioral_weights'):
            weights = self.sim.behavioral_weights
            
            # Create sliders for each weight (always use sliders for clarity)
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
            # Gather attributes present on the weights object
            present_attrs = [a for a in weight_attrs if hasattr(weights, a)]
            # If many attrs, present them in a scroll area with explicit sliders (simpler and clearer)
            if len(present_attrs) > 10:
                scroll = QtWidgets.QScrollArea()
                scroll.setWidgetResizable(True)
                inner = QWidget()
                inner_layout = QVBoxLayout()
                inner_layout.setSpacing(4)
                inner_layout.setContentsMargins(2, 2, 2, 2)
                for attr in present_attrs:
                    try:
                        value = getattr(weights, attr)
                    except Exception:
                        value = 0.0

                    row = QHBoxLayout()
                    lbl = QLabel(f"{attr.replace('_',' ').title()}: ")
                    lbl.setStyleSheet("font-size:9pt; color:black;")
                    row.addWidget(lbl)

                    if isinstance(value, bool):
                        cb = QCheckBox()
                        cb.setChecked(bool(value))
                        cb.stateChanged.connect(lambda s, a=attr, cb=cb: self.update_bool_weight(a, cb.isChecked()))
                        row.addWidget(cb)
                        self.weight_sliders[attr] = cb
                    else:
                        slider = QSlider(Qt.Horizontal)
                        slider.setFixedHeight(18)
                        slider.setStyleSheet("QSlider::handle:horizontal{background:#3b99fc;border-radius:6px;width:10px;height:10px;} QSlider{margin:2px 4px;}")
                        # default mapping
                        slider_min, slider_max = 0, 200
                        slider_value = int(float(value) * 100)
                        slider.setMinimum(slider_min)
                        slider.setMaximum(slider_max)
                        slider.setValue(max(slider_min, min(slider_max, slider_value)))
                        slider.valueChanged.connect(lambda v, a=attr, l=lbl: self.update_weight(a, v, l))
                        row.addWidget(slider)
                        self.weight_sliders[attr] = slider

                    inner_layout.addLayout(row)

                inner.setLayout(inner_layout)
                scroll.setWidget(inner)
                layout.addWidget(scroll)

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

        # --- Per-episode metrics group (plot + checkboxes to control plotted series) ---
        per_episode_group = QGroupBox("Per-Episode Metrics")
        pe_layout = QVBoxLayout()

        # Available metrics that can be tracked per-episode
        self._available_episode_metrics = [
            'collision_count', 'mean_upstream_progress', 'mean_upstream_velocity',
            'energy_efficiency', 'mean_passage_delay'
        ]

        # Compact row of checkboxes to toggle which metrics appear on the per-episode plot
        cb_row = QHBoxLayout()
        cb_row.setSpacing(8)
        self.per_episode_checkboxes = {}
        for m in self._available_episode_metrics:
            cb = QCheckBox(m.replace('_', ' ').title())
            cb.setChecked(False)
            cb.stateChanged.connect(self._on_per_episode_cb_changed)
            self.per_episode_checkboxes[m] = cb
            cb_row.addWidget(cb)
        pe_layout.addLayout(cb_row)

        # per-episode plot (shows one point per episode per tracked metric)
        self.per_episode_plot = pg.PlotWidget(title="Per-Episode Metrics")
        self.per_episode_plot.setLabel('bottom', 'Episode')
        self.per_episode_plot.setLabel('left', 'Metric Value')
        self.per_episode_plot.setMaximumHeight(220)
        pe_layout.addWidget(self.per_episode_plot)

        self.per_episode_series = {m: [] for m in self._available_episode_metrics}
        self.per_episode_handles = {}

        per_episode_group.setLayout(pe_layout)
        layout.addWidget(per_episode_group)

        weights_group.setLayout(layout)
        return weights_group
    
    def create_metrics_panel(self):
        """Create a basic metrics panel for the left sidebar."""
        panel = QGroupBox("Metrics")
        layout = QVBoxLayout()
        self.mean_speed_label = QLabel("Mean Speed: --")
        self.max_speed_label = QLabel("Max Speed: --")
        self.mean_energy_label = QLabel("Mean Energy: --")
        self.min_energy_label = QLabel("Min Energy: --")
        layout.addWidget(self.mean_speed_label)
        layout.addWidget(self.max_speed_label)
        layout.addWidget(self.mean_energy_label)
        layout.addWidget(self.min_energy_label)
        panel.setLayout(layout)
        return panel
    
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
            # Accumulate per-episode values based on the per-episode checkboxes
            if not hasattr(self, 'episode_metric_accumulators'):
                self.episode_metric_accumulators = {}
            for m, cb in getattr(self, 'per_episode_checkboxes', {}).items():
                try:
                    if cb.isChecked() and m in metrics:
                        if m not in self.episode_metric_accumulators:
                            self.episode_metric_accumulators[m] = []
                        self.episode_metric_accumulators[m].append(float(metrics[m]))
                except Exception:
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

    def _on_per_episode_cb_changed(self):
        """Rebuild per-episode plotted series when user toggles checkboxes."""
        try:
            # Clear existing handles
            for h in self.per_episode_handles.values():
                try:
                    self.per_episode_plot.removeItem(h)
                except Exception:
                    pass
            self.per_episode_handles = {}
            # Recreate handles for checked metrics
            colors = [(255, 200, 0), (0, 200, 255), (200, 100, 255), (100, 255, 100), (180, 180, 255)]
            for i, m in enumerate(self._available_episode_metrics):
                cb = self.per_episode_checkboxes.get(m)
                if cb is not None and cb.isChecked():
                    series = self.per_episode_series.get(m, [])
                    pen = pg.mkPen(color=colors[i % len(colors)], width=2)
                    handle = self.per_episode_plot.plot(list(range(len(series))), series, pen=pen, name=m)
                    self.per_episode_handles[m] = handle
        except Exception:
            pass

    def _refresh_per_episode_handles_after_append(self):
        """Update per-episode handles incrementally when new episode points are appended."""
        try:
            for m, handle in list(self.per_episode_handles.items()):
                series = self.per_episode_series.get(m, [])
                try:
                    handle.setData(list(range(len(series))), series)
                except Exception:
                    # recreate handle if needed
                    del self.per_episode_handles[m]
            # create handles for any newly checked series without handles
            for m, cb in self.per_episode_checkboxes.items():
                if cb.isChecked() and m not in self.per_episode_handles:
                    series = self.per_episode_series.get(m, [])
                    pen = pg.mkPen(width=2)
                    handle = self.per_episode_plot.plot(list(range(len(series))), series, pen=pen, name=m)
                    self.per_episode_handles[m] = handle
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

    def update_speed(self, value):
        """Update simulation speed."""
        # value ranges 1-100, map to 0.1x - 10x
        speed = value / 10.0
        self.speed_label.setText(f"Speed: {speed:.1f}x")
        
        # Update timer interval
        interval = int(self.dt * 1000 / speed)
        self.timer.setInterval(max(1, interval))

    def update_vert_exag_label(self, value):
        """Update the vertical exaggeration label when the slider moves."""
        ve = value / 100.0
        try:
            self.ve_label.setText(f"Z Exag: {ve:.2f}x")
        except Exception:
            pass

    def rebuild_tin_action(self):
        """Handler for 'Rebuild TIN' button: re-run TIN rendering with current VE."""
        try:
            ve = self.ve_slider.value() / 100.0
            setattr(self.sim, 'vert_exag', ve)
            # If HECRAS-based geometry is present, use thinned coords/vals if stored
            # We attempt to reuse last thinned arrays if available (set by setup_background())
            if hasattr(self, 'last_thinned_coords') and hasattr(self, 'last_thinned_vals'):
                self.render_tin_from_arrays(self.last_thinned_coords, self.last_thinned_vals, cap_nodes=getattr(self.sim, 'tin_max_nodes', 5000), vert_exag=ve)
            else:
                # fallback: re-run full background setup which will compute thinned arrays and call render
                self.setup_background()
        except Exception as e:
            print('Rebuild TIN failed:', e)
    
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
                    # refresh handles if necessary
                    try:
                        self._refresh_per_episode_handles_after_append()
                    except Exception:
                        pass
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

        # Draw purely-visual two-segment tails (proximal rigid + distal oscillating)
        if self.show_tail_cb.isChecked() and hasattr(self.sim, 'heading'):
            # clear old tail lines
            if not hasattr(self, 'tail_lines'):
                self.tail_lines = []
            for line in getattr(self, 'tail_lines', []):
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass
            self.tail_lines = []

            # Determine visible agents
            if self.show_dead_cb.isChecked():
                headings = self.sim.heading
                pos_x, pos_y = self.sim.X, self.sim.Y
            else:
                alive_mask = (self.sim.dead == 0)
                headings = self.sim.heading[alive_mask]
                pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]

            # Compute current time (seconds)
            current_time = getattr(self, 'current_timestep', 0) * float(getattr(self, 'dt', 1.0))

            # For each agent, compute two segments
            nvis = len(pos_x)
            # ensure phases length
            phases = self.tail_phases
            if len(phases) != self.sim.num_agents:
                phases = np.resize(phases, self.sim.num_agents)

            for i in range(nvis):
                # global agent index mapping when filtering dead agents
                if self.show_dead_cb.isChecked():
                    gidx = i
                else:
                    # find i-th True in alive_mask
                    alive_idx = np.nonzero(alive_mask)[0]
                    if i < len(alive_idx):
                        gidx = int(alive_idx[i])
                    else:
                        gidx = i

                h = float(headings[i])
                x0 = float(pos_x[i])
                y0 = float(pos_y[i])

                # Proximal rigid segment (L1) aligned with heading (trailing behind fish)
                x1 = x0 - self.tail_L1 * np.cos(h)
                y1 = y0 - self.tail_L1 * np.sin(h)

                # Distal oscillating segment (L2) pivoting about (x1,y1)
                phase = float(phases[gidx]) if gidx < len(phases) else 0.0
                theta = self.tail_amp * np.sin(2.0 * np.pi * self.tail_freq * current_time + phase)
                # distal orientation = heading + theta (relative oscillation)
                x2 = x1 - self.tail_L2 * np.cos(h + theta)
                y2 = y1 - self.tail_L2 * np.sin(h + theta)

                # Draw proximal (thicker) and distal (thinner, brighter)
                try:
                    seg1 = self.plot_widget.plot([x0, x1], [y0, y1], pen=pg.mkPen(color=(180, 180, 255, 200), width=2))
                    seg2 = self.plot_widget.plot([x1, x2], [y1, y2], pen=pg.mkPen(color=(255, 220, 180, 220), width=1))
                    self.tail_lines.append(seg1)
                    self.tail_lines.append(seg2)
                except Exception:
                    pass
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
