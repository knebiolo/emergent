# -*- coding: utf-8 -*-
# ship_viewer.py

import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem
from PyQt5.QtGui     import QPolygonF, QBrush, QPen, QColor
from PyQt5.QtCore    import QPointF
from emergent.ship_abm.simulation_core import simulation, SIMULATION_BOUNDS
from emergent.ship_abm.simulation_core import playful_wind, north_south_current
from emergent.ship_abm.simulation_core import compute_zigzag_metrics
from datetime import date

class ship_polygon(QGraphicsPolygonItem):
    """
    Simple QGraphicsItem that draws a filled, edged polygon from an Nx2 array.
    You can move it with setPos(x,y) and rotate with setRotation(degrees).
    """
    def __init__(self, body: np.ndarray,
                 color=(200,50,50,180),
                 pen_color=(0,0,0),
                 pen_width=0.3):
        super().__init__()
        # build the QPolygonF
        pts = [QPointF(float(x), float(y)) for x,y in body]
        self.setPolygon(QPolygonF(pts))
        # fill
        self.setBrush(QBrush(QColor(*color)))
        # outline
        pen = QPen(QColor(*pen_color))
        pen.setWidthF(pen_width)
        self.setPen(pen)

class ship_viewer(QtWidgets.QWidget):
    """
    A Qt‐based viewer that draws the ENC basemap (coastlines and shoals) in UTM,
    using PyQtGraph.  Ships, routes, and buttons can be added incrementally.
    """
    def __init__(self,
                 port_name: str,
                 xml_url: str,
                 dt: float = 0.1,
                 T: float = 100,
                 n_agents: int = 0,
                 coast_simplify_tol: float = 50.0,
                 light_bg: bool = True,
                 verbose: bool = False,
                 use_ais: bool = False,
                 # ---- test-mode passthrough ----
                 test_mode: str | None = None,
                 zigzag_deg: int = 10,
                 zigzag_hold: float = 40.0):
        
        super().__init__()
        self.setWindowTitle(f"Ship ABM Viewer – {port_name}")
        self.resize(1200, 800)
        self.ais_removed = False
        self.use_ais = use_ais
        self.ais_item = None
        # ─────────────────────────────────────────────────────────────────────
        # 1) Instantiate the headless core (which loads ENC data)
        # ─────────────────────────────────────────────────────────────────────
        self.sim = simulation(
            port_name=port_name,
            dt=dt,
            T=T,
            n_agents=n_agents,
            coast_simplify_tol=coast_simplify_tol,
            light_bg=light_bg,
            verbose=verbose,
            use_ais=use_ais,
            test_mode=test_mode,         
            zigzag_deg=zigzag_deg,
            zigzag_hold=zigzag_hold
        )

        # Assure that ENC features are loaded (if not done in core __init__)
        # (In your core, load_enc_features should have populated self.sim.enc_data)
        # Example: self.sim.load_enc_features(xml_url, verbose=verbose)

        # ─────────────────────────────────────────────────────────────────────
        # 2) Set up PyQtGraph GraphicsLayoutWidget
        # ─────────────────────────────────────────────────────────────────────
        layout = QtWidgets.QHBoxLayout(self)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view = self.plot_widget.addViewBox()
        self.view.setAspectLocked(True)   # lock 1:1 scaling

        # ── visual tweaks for zig-zag test mode ────────────────────
        if test_mode == "zigzag":
            self.plot_widget.setBackground("w")               # white bg
            grid = pg.GridItem(pen=pg.mkPen((200, 200, 200), width=0.6))
            grid.setZValue(-10)
            self.view.addItem(grid)
        layout.addWidget(self.plot_widget, stretch=4)

        # Placeholder on the right for controls / buttons and graphics panel
        ctrl_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl_layout, stretch=1)
        # graphics panel for compass rose and vanes
        self.panel_widget = pg.GraphicsLayoutWidget()
        ctrl_layout.addWidget(self.panel_widget)
        self.panel = self.panel_widget.addViewBox()
        self.panel.setAspectLocked(True)
        self.panel.setMouseEnabled(False, False)
        # grey background for vane panel
        self.panel.setBackgroundColor(QBrush(QColor(200,200,200)))
        # set fixed range so vanes at (40,40) and (40,100) appear
        from PyQt5.QtCore import QRectF
        self.panel.setRange(QRectF(0, 0, 200, 200), padding=0)
        ctrl_layout.addStretch()
        

        # ─────────────────────────────────────────────────────────────────────
        #  ENVIRONMENTAL PANEL
        # ─────────────────────────────────────────────────────────────────────
        self.lbl_time    = QtWidgets.QLabel("Time: 0.00 s")
        self.lbl_wind    = QtWidgets.QLabel("Wind: 0.00 m/s, 0°")
        self.lbl_current = QtWidgets.QLabel("Current: 0.00 m/s, 0°")
        for lbl in (self.lbl_time, self.lbl_wind, self.lbl_current):
            ctrl_layout.addWidget(lbl)

        # Wind-vane arrow in side panel
        self.wind_arrow = pg.ArrowItem(angle=0, tipAngle=90, baseAngle=45, headLen=20)
        self.wind_arrow.setZValue(50)
        self.panel.addItem(self.wind_arrow)
        self.wind_arrow.setPos(40, 40)
        
        # Current-vane arrow in side panel
        self.current_arrow = pg.ArrowItem(angle=0, tipAngle=90, baseAngle=45, headLen=20)
        self.current_arrow.setZValue(50)
        self.panel.addItem(self.current_arrow)
        self.current_arrow.setPos(40, 100)

        # ─────────────────────────────────────────────────────────────────────
        # 3) Draw the basemap only if ENC layers were actually loaded
        if self.sim.enc_data:             # empty => skipped in test mode
            self._draw_basemap()

        # ───────────────────────────────────────────────────────────────────          
        # ROUTE widgets only when routing is relevant
        if test_mode != "zigzag":
            self.btn_define_route = QtWidgets.QPushButton("Define Route")
            self.btn_define_route.clicked.connect(self._start_route_mode)
            if hasattr(self.sim, "use_ais") and hasattr(self.sim, "ais_extent"):
                from pyqtgraph import ImageItem
                img = ImageItem(self.sim.ais_heatmap.T)
                img.setRect(QtCore.QRectF(*self.sim.ais_extent))
                img.setOpacity(0.35)
                self.view.addItem(img)
                self.ais_item = img  # so we can later remove it in simulation mode
            self.btn_define_route.setToolTip(
                "Left-click: add waypoint\n"
                "Right-click (once ≥2 points): finish route"
            )
            ctrl_layout.addWidget(self.btn_define_route)

            self._status_label = QtWidgets.QLabel("")
            self._status_label.setStyleSheet("color: black;")
            ctrl_layout.addWidget(self._status_label)


        # Done with initialization
        self.show()
        
        # ----- per-agent routing state -----
        # Initialize waypoint buffers; click handling starts in route mode
        if test_mode != "zigzag":
            # ── Overlay AIS heatmap ─────────────────────────────────────

            self._route_pts_by_agent = [[] for _ in range(self.sim.n)]
            self._current_route_agent = 0
            self._status_label.setText(f"Draw route for agent 1 of {self.sim.n}")
            self._route_pts = []  # temporary holder for current agent's clicks
        # NOTE: do NOT connect sigMouseClicked here; handled in _start_route_mode

        # Usage instructions label (skip in zig-zag)
        if test_mode != "zigzag":
            
            instr = QtWidgets.QLabel(
                "▶ Define Route:\n"
                "   • Left-click on map to add points\n"
                "   • Right-click to finish (needs ≥2 points)\n"
                "\n▶ Start Simulation when ready"
            )
            instr.setWordWrap(True)
            instr.setStyleSheet("color: black;")
            ctrl_layout.addWidget(instr)
        
        # START button
        self.btn_start_sim = QtWidgets.QPushButton("Start Simulation")
        self.btn_start_sim.clicked.connect(self._start_simulation)
        ctrl_layout.addWidget(self.btn_start_sim)
        if test_mode == "zigzag":
            self.btn_start_sim.setEnabled(True)   # auto-enabled
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frame)

        # ─────────────────────────────────────────────────────────────────────
        # 5) Placeholder lists for ship‐polygon items, route items, etc.
        # ─────────────────────────────────────────────────────────────────────
        self.ship_items = []    # will hold custom polygon graphics items
        self.route_item = None  # will hold the planned route (PlotCurveItem)
        #self.ais_item = None    # will hold an AIS heatmap (ImageItem)

        # ─────────────────────────────────────────────────────────────────────
        #   Annotation items (per‐frame)
        # ─────────────────────────────────────────────────────────────────────
        self.rudder_items  = []  # yellow lines showing rudder deflection
        self.heading_items = []  # red arrows for desired heading
        self.label_items   = []  # text labels with hd_cmd, hd_cur, err, rud

    def _start_simulation(self):
        """
        Called when the user clicks 'Start Simulation'.
        Spawns agents in the core, draws ship polygons, and kicks off the timer.
        """
        print("▶ _start_simulation called")
        
        # Hide AIS layer and remove plotted route
        if self.ais_item is not None:
            try:
                self.view.removeItem(self.ais_item)
            except Exception:
                pass
            self.ais_item = None
        if self.route_item is not None:
            try:
                self.view.removeItem(self.route_item)
            except Exception:
                pass
            self.route_item = None
            
        # 1) spawn in the core
        try:
            state0, pos0, psi0, goals = self.sim.spawn()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Spawn Error", str(e))
            return
        print(f"   state0={state0.shape}, pos0={pos0[:,0]}, psi0={psi0[0]}")
        
        # 2) save state
        self.sim.pos   = pos0
        self.sim.psi   = psi0
        self.sim.goals = goals

        # 3) draw ship polygons
        self.ship_items = []
        for i in range(self.sim.n):
            body = self.sim._ship_base
            poly = ship_polygon(body, color=(200,50,50,180))
            poly.setPos(pos0[0, i], pos0[1, i])
            poly.setRotation(np.degrees(psi0[i]))
            poly.setZValue(30)
            self.view.addItem(poly)
            self.ship_items.append(poly)
            
        x0, y0 = pos0[0,0], pos0[1,0]
        width = height = self.sim.zoom  # e.g. 5000m
        rect = QtCore.QRectF(x0 - width/2, y0 - height/2, width, height)
        self.view.setRange(rect, padding=0.02)
        print(f"   view centered at {x0:.0f},{y0:.0f} ±{width/2}")
        
        # 4) start the animation timer
        self.timer.start(int(self.sim.dt * 1000))
        self.btn_start_sim.setEnabled(False)
        self.btn_define_route.setEnabled(False)

    def _draw_ais(self, start_date, end_date, cell_size=100.0):
        """
        Overlay AIS density on self.view with a colored LUT and 
        ensure trajectories render above it.
        """
        # 1) choose grid so each cell ~cell_size meters
        xmin, xmax = self.sim.minx, self.sim.maxx
        ymin, ymax = self.sim.miny, self.sim.maxy
        nx = max(1, int((xmax - xmin) / cell_size))
        ny = max(1, int((ymax - ymin) / cell_size))

        # 2) fetch the raw counts
        arr, (x0, x1, y0, y1) = self.sim.get_ais_heatmap(
            date_range=(start_date, end_date),
            grid_size=(nx, ny)
        )
        print(f"[AIS] binned into {nx}×{ny} cells → total={arr.sum()}, max={arr.max()}")

        # 3) create the ImageItem
        img = pg.ImageItem(arr.T, opacity=0.6)
        img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))

        # 4) build or load a LUT
        try:
            # 'viridis' is bundled with pyqtgraph
            cmap = pg.colormap.get('viridis')
            lut = cmap.getLookupTable(0.0, 1.0, 256)
        except Exception:
            # fallback to a simple blue→red gradient
            lut = np.zeros((256, 4), dtype=np.uint8)
            lut[:, 0] = np.linspace(0, 255, 256)   # red up
            lut[:, 2] = np.linspace(255, 0, 256)   # blue down
            lut[:, 3] = 255                         # opaque
        img.setLookupTable(lut)

        # 5) clamp levels for contrast (0 → 99th percentile)
        img.setLevels([0, np.percentile(arr, 99)])

        # 6) push it behind everything else
        img.setZValue(-10)

        self.view.addItem(img)
        self.ais_item = img

    def _draw_basemap(self):
        """
        Draw a vector navigation chart background:
        - Water = light‐blue
        - Land areas (LNDARE) = tan fill + goldenrod outline
        - Shoals (DEPARE) = pale‐blue fill + darker outline (if light_bg=False)
        """
        # ── Plot UTM graticules ────────────────────────────────
        spacing = 1000  # meters
        x_ticks = np.arange(self.sim.minx, self.sim.maxx + spacing, spacing)
        y_ticks = np.arange(self.sim.miny, self.sim.maxy + spacing, spacing)
        
        for x in x_ticks:
            line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen((80, 80, 80, 150), width=0.5))
            self.view.addItem(line)
        for y in y_ticks:
            line = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen((80, 80, 80, 150), width=0.5))
            self.view.addItem(line)
            
        # 1) Water background
        self.plot_widget.setBackground(QColor(173, 216, 230))  # light‐blue

        # 2) Land areas
        land_gdf = self.sim.enc_data.get("LNDARE")
        if land_gdf is not None and not land_gdf.empty:
            for geom in land_gdf.geometry:
                if isinstance(geom, Polygon):
                    polys = [geom]
                elif isinstance(geom, MultiPolygon):
                    polys = geom.geoms
                else:
                    continue
                for poly in polys:
                    coords = np.array(poly.exterior.coords)
                    qpts = [QPointF(x, y) for x, y in coords]
                    qp = QPolygonF(qpts)
                    item = QGraphicsPolygonItem(qp)
                    # tan fill
                    item.setBrush(QBrush(QColor(205, 185, 135)))  # tan
                    # goldenrod outline
                    item.setPen(QPen(QColor(218, 165, 32), 3.0))
                    item.setZValue(0)
                    self.view.addItem(item)

        # 3) Shoals (if requested)
        if not self.sim.light_bg:
            shoal_gdf = self.sim.enc_data.get("DEPARE")
            if shoal_gdf is not None and not shoal_gdf.empty:
                for geom in shoal_gdf.geometry:
                    if isinstance(geom, Polygon):
                        polys = [geom]
                    elif isinstance(geom, MultiPolygon):
                        polys = geom.geoms
                    else:
                        continue
                    for poly in polys:
                        coords = np.array(poly.exterior.coords)
                        qpts = [QPointF(x, y) for x, y in coords]
                        qp = QPolygonF(qpts)
                        item = QGraphicsPolygonItem(qp)
                        # pale‐blue fill matching matplotlib
                        item.setBrush(QBrush(QColor(177, 255, 1, 100)))
                        # lighter blue outline matching matplotlib
                        item.setPen(QPen(QColor(100, 149, 237), 0.4))
                        item.setZValue(1)
                        self.view.addItem(item)

        # # 4) Coastline outlines (if you want extra emphasis)
        # #    You can re‐iterate land boundaries or use COALNE layer:
        # coast_gdf = self.sim.enc_data.get("COALNE")
        # if coast_gdf is not None and not coast_gdf.empty:
        #     pen = pg.mkPen(color=(218, 165, 32), width=1.0)  # goldenrod
        #     for geom in coast_gdf.geometry:
        #         if geom.geom_type == "LineString":
        #             coords = np.array(geom.coords)
        #             self.view.addItem(pg.PlotCurveItem(
        #                 x=coords[:,0], y=coords[:,1], pen=pen
        #             ))
        #         elif geom.geom_type == "MultiLineString":
        #             for line in geom.geoms:
        #                 coords = np.array(line.coords)
        #                 self.view.addItem(pg.PlotCurveItem(
        #                     x=coords[:,0], y=coords[:,1], pen=pen
        #                 ))

        # 5) Finally, set the view range to the port bounds
        x0, x1 = self.sim.minx, self.sim.maxx
        y0, y1 = self.sim.miny, self.sim.maxy
        self.view.setRange(QtCore.QRectF(x0, y0, x1-x0, y1-y0), padding=0.02)
        
        # 6) Depth contours (DEPVAL)
        contours = self.sim.enc_data.get("DEPVAL")
        if contours is not None and not contours.empty:
            pen = pg.mkPen(color=(0,0,139), width=5., style=QtCore.Qt.SolidLine)  # thicker, solid navy blue
            for geom in contours.geometry:
                lines = [geom] if geom.geom_type=="LineString" else geom.geoms
                for line in lines:
                    coords = np.array(line.coords)
                    self.view.addItem(pg.PlotCurveItem(x=coords[:,0], y=coords[:,1], pen=pen))
                    
        # Depth contour labels (if available)
        depth_labels = self.sim.enc_data.get("DRVAL2")
        if depth_labels is not None and not depth_labels.empty:
            for _, row in depth_labels.iterrows():
                val = row.get('DRVAL2', None)
                if val is None:
                    continue
                pt = row.geometry.representative_point()
                lbl = pg.TextItem(text=f"{val}", color=(0,0,139))
                lbl.setPos(pt.x(), pt.y())
                lbl.setZValue(100)
                self.view.addItem(lbl)
                    
        # 7) Bridge infrastructure (if loaded)
        bridges = self.sim.enc_data.get("BRIDGE")
        if bridges is not None and not bridges.empty:
            for geom in bridges.geometry:
                coords = np.array(geom.exterior.coords) if hasattr(geom, 'exterior') else np.array(geom.coords)
                poly = QPolygonF([QPointF(x, y) for x, y in coords])
                item = QGraphicsPolygonItem(poly)
                item.setBrush(QBrush(QColor(150,150,150,150)))
                item.setPen(QPen(QColor(80,80,80), 1))
                item.setZValue(48)
                self.view.addItem(item)
                
        # 8) Channel markers (if present)
        markers = self.sim.enc_data.get("BOY")  # ENC buoy/marker layer
        if markers is not None and not markers.empty:
            # outgoing left = CAN (green square), incoming right = NUN (red triangle)
            can = markers[markers.get("TOPSHP") == "CAN"]
            if not can.empty:
                xs = list(can.geometry.x)
                ys = list(can.geometry.y)
                sp_can = pg.ScatterPlotItem(x=xs, y=ys, symbol='s', size=12,
                                            brush=pg.mkBrush(0,255,0), pen=None)
                sp_can.setZValue(100)
                self.view.addItem(sp_can)
            nun = markers[markers.get("TOPSHP") == "NUN"]
            if not nun.empty:
                xs = list(nun.geometry.x)
                ys = list(nun.geometry.y)
                sp_nun = pg.ScatterPlotItem(x=xs, y=ys, symbol='t', size=12,
                                            brush=pg.mkBrush(255,0,0), pen=None)
                sp_nun.setZValue(100)
                self.view.addItem(sp_nun)

        # 9) Obstructions (if present)
        obstrs = self.sim.enc_data.get("OBSTRN")
        if obstrs is not None and not obstrs.empty:
            # point obstructions as red circles
            # pts = obstrs[obstrs.geometry.type == "Point"]
            # if not pts.empty:
            #     xs = pts.geometry.x.tolist()
            #     ys = pts.geometry.y.tolist()
            #     sp_obs = pg.ScatterPlotItem(x=xs, y=ys, symbol='o', size=8,
            #                                 brush=pg.mkBrush(255,0,0), pen=None)
            #     sp_obs.setZValue(5)
            #     self.view.addItem(sp_obs)
            # polygonal obstructions as semi-transparent red polygons
            polys = obstrs[obstrs.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not polys.empty:
                for geom in polys.geometry:
                    poly_list = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                    for poly in poly_list:
                        coords = np.array(poly.exterior.coords)
                        qpts = [QPointF(x, y) for x, y in coords]
                        obs_item = QGraphicsPolygonItem(QPolygonF(qpts))
                        obs_item.setBrush(QBrush(QColor(255, 0, 0, 80)))
                        obs_item.setPen(QPen(QColor(255, 0, 0), 1))
                        obs_item.setZValue(5)
                        self.view.addItem(obs_item)
                    
    def _on_map_click(self, ev):
        button = ev.button()
        if button == QtCore.Qt.LeftButton:
            # LEFT‐click: add a waypoint for the current agent
            pt = self.view.mapSceneToView(ev.scenePos())
            x, y = float(pt.x()), float(pt.y())
            self._route_pts_by_agent[self._current_route_agent].append((x, y))
            dot = pg.ScatterPlotItem([x], [y], symbol='o',
                                     pen=pg.mkPen('m'), brush=pg.mkBrush('m'), size=8)
            dot.setZValue(20)
            self.view.addItem(dot)
        elif button == QtCore.Qt.RightButton:
            # RIGHT‐click: finalize this agent's route
            pts = self._route_pts_by_agent[self._current_route_agent]
            if len(pts) < 2:
                return
            # stop listening for this agent
            try:
                self.view.scene().sigMouseClicked.disconnect(self._on_map_click)
            except TypeError:
                pass
            # if more agents remain, advance to next
            if self._current_route_agent < self.sim.n - 1:
                self._current_route_agent += 1
                self._status_label.setText(
                    f"Draw route for agent {self._current_route_agent+1} of {self.sim.n}"
                )
                # reconnect for next agent's clicks
                self.view.scene().sigMouseClicked.connect(self._on_map_click)
            else:
                # all routes defined → hand off to simulation
                self.sim.waypoints = [
                    [np.array(p) for p in agent_pts]
                    for agent_pts in self._route_pts_by_agent
                ]
                self.sim.route()
                # draw each agent's path
                for agent_pts in self._route_pts_by_agent:
                    xs, ys = zip(*agent_pts)
                    curve = pg.PlotCurveItem(
                        x=np.array(xs), y=np.array(ys),
                        pen=pg.mkPen('lime', width=2)
                    )
                    curve.setZValue(15)
                    self.view.addItem(curve)
                self.btn_start_sim.setEnabled(True)
                self._status_label.setText("All routes set — ready to simulate.")


    def _start_route_mode(self):
        """Enable click‑to‑collect‑waypoints mode for per-agent routing."""
        # Reset routing state
        self._route_pts_by_agent = [[] for _ in range(self.sim.n)]
        self._current_route_agent = 0
        self.btn_define_route.setEnabled(False)
        self.btn_start_sim.setEnabled(False)
        if self.use_ais:
            self._draw_ais(start_date=date(2024, 6, 1), end_date=date(2024, 6, 2))

        # Prompt first agent
        self._status_label.setText(f"Draw route for agent 1 of {self.sim.n}")
        # listen for map clicks to collect waypoints
        self.view.scene().sigMouseClicked.connect(self._on_map_click)

    def _update_frame(self):
        """
        Called by QTimer every dt seconds.
        Advances the simulation and moves ship polygons.
        """
        # ── One-shot AIS removal ──────────────────────────────────────────
        if not self.ais_removed and getattr(self, 'ais_item', None) is not None:
            try:
                self.view.removeItem(self.ais_item)
            except Exception:
                pass
            self.ais_item = None
            self.ais_removed = True

        # 1) stop if requested
        if getattr(self.sim, "_stop_flag", False) or self.sim.t >= self.sim.steps * self.sim.dt:
            self.timer.stop()
            # compute and print metrics when GUI test completes
            t_arr  = np.array(self.sim.t_history)
            psi_arr= np.array(self.sim.psi_history)
            hd_arr = np.array(self.sim.hd_cmd_history)
            metrics = compute_zigzag_metrics(t_arr, psi_arr, hd_arr)
            print("Zig-zag metrics:", metrics)
        
            # simulation complete: close window and quit Qt loop
            try:
                from PyQt5 import QtWidgets
                QtWidgets.QApplication.quit()
            except Exception:
                pass
            return
        # 2)  update dynamic waypoint goals (pop reached waypoints)
        self.sim._update_goals()

        # 3)  compute controls & integrate one step
        #    (match the core’s run() loop)
        t = self.sim.t
        # pack nu = [u; v; r] rows
        nu = np.vstack([
            self.sim.state[0],   # surge
            self.sim.state[1],   # sway
            self.sim.state[3]    # yaw rate
        ])
        # compute heading, speed, rudder commands (arrays of length n)
        hd_cmds, sp_cmds, rud_cmds = self.sim._compute_controls_and_update(
            self.sim.state[[0,1,3]], t)   # pass a view; no fresh vstack

        # ── DEBUG ───────────────────────────────────────────────────────────────
        # Extract the primary agent’s values as scalars
        hd0 = float(hd_cmds.flat[0])
        rud0 = float(rud_cmds.flat[0])
        curr_hd_deg = float(np.degrees(self.sim.psi[0]))
        err0 = ((np.degrees(hd0) - curr_hd_deg + 180) % 360) - 180
        print(
            f"[CTRL] t={t:.2f}s → "
            f"hd_cmd={np.degrees(hd0):.1f}°, "
            f"hd_cur={curr_hd_deg:.1f}°, "
            f"err={err0:.1f}°, rud={rud0:.3f}"
        )

        # step dynamics with those commands
        self.sim._step_dynamics(hd_cmds, sp_cmds, rud_cmds)
        # ── Turning circle override ────────────────────────────────
        if getattr(self, 'test_mode', None) == "turncircle":
            rud_cmds = np.full(self.n, getattr(self, 'constant_rudder_cmd', np.deg2rad(20.0)))

        # advance the time
        self.sim.t += self.sim.dt
        # record history for zigzag metrics
        self.sim.t_history.append(self.sim.t)
        self.sim.psi_history.append(self.sim.psi[0])
        self.sim.hd_cmd_history.append(hd_cmds[0])


        # 3) update each ship polygon’s position & rotation
        for i, poly in enumerate(self.ship_items):
            x, y = self.sim.pos[0, i], self.sim.pos[1, i]
            psi_i = self.sim.psi[i]
            poly.setPos(x, y)
            poly.setRotation(np.degrees(psi_i))
            
        print(f"▶ tick t={self.sim.t:.2f}, pos={self.sim.pos[:,0] if self.sim.pos is not None else 'N/A'}")

        # ── Clear previous annotations ─────────────────────────────────────
        for lst in (self.rudder_items, self.heading_items):
            for itm in lst:
                self.view.removeItem(itm)
            lst.clear()

        # ── DRAW RUDDER LINES, HEADING ARROWS & TEXT LABELS ──────────────
        L = self.sim.ship.length
        # ── make sure we have one TextItem per ship (one-time) ─────────────
        need = self.sim.n - len(self.label_items)
        for _ in range(need):
            itm = pg.TextItem(anchor=(0, 1))          # lower-left anchor
            itm.setZValue(100)
            # ── force text colour depending on mode ─────────────────
            if getattr(self.sim, "test_mode", None) == "zigzag":
                itm.setColor('k')      # black for white bg
            else:
                itm.setColor('w')      # default white for dark ENC bg          
            self.view.addItem(itm)
            self.label_items.append(itm)

        # ── gather arrays once (avoid per-ship slicing) ────────────────────
        u_arr, v_arr, r_arr = self.sim.state[0], self.sim.state[1], self.sim.state[3]
        rpm_cmd   = self.sim.ship.commanded_rpm
        rud_deg   = np.degrees(self.sim.ship.smoothed_rudder)
        roles     = getattr(self.sim.ship, "_last_role",
                            ["neutral"] * self.sim.n)   # graceful fallback
        thr_pct   = 100.0 * np.atleast_1d(rpm_cmd) / self.sim.ship.max_rpm

        for i in range(self.sim.n):
            x, y    = self.sim.pos[0,i], self.sim.pos[1,i]
            curr_hd = self.sim.psi[i]
            cmd_hd  = hd_cmds[i]
            rud_i   = rud_cmds[i]

            # 1) Rudder line at the stern (length=0.1·L, tilt=ψ+π+δ+signδ·π/2)
            stern_offset = (L/2) + 0.05 * L
            stern_x = x - stern_offset * np.cos(curr_hd)
            stern_y = y - stern_offset * np.sin(curr_hd)
            tilt     = np.sign(rud_i) * (np.pi/2)
            rud_end_x = stern_x - 0.1 * L * np.cos(curr_hd + np.pi + rud_i + tilt)
            rud_end_y = stern_y - 0.1 * L * np.sin(curr_hd + np.pi + rud_i + tilt)
            rud_line = pg.PlotCurveItem(
                x=[stern_x, rud_end_x], y=[stern_y, rud_end_y],
                pen=pg.mkPen('yellow', width=2)
            )
            rud_line.setZValue(100)  # draw above hull
            self.view.addItem(rud_line)
            self.rudder_items.append(rud_line)


            # 2) Desired-heading line at the bow (length=0.2·L)
            bow_x = x + (L/2) * np.cos(cmd_hd)
            bow_y = y + (L/2) * np.sin(cmd_hd)
            dx_hd = 0.2 * L * np.cos(cmd_hd)
            dy_hd = 0.2 * L * np.sin(cmd_hd)
            hd_line = pg.PlotCurveItem(
                x=[bow_x, bow_x + dx_hd], y=[bow_y, bow_y + dy_hd],
                pen=pg.mkPen('red', width=2)
            )
            self.view.addItem(hd_line)
            self.heading_items.append(hd_line)

            # 3) Text label --------------------------------------------------
            r_dps = np.degrees(r_arr[i])        # deg s-¹ is what mariners think in

            
            # choose font colour & size based on mode
            colour = "black" if getattr(self.sim, "test_mode", None) == "zigzag" else "white"
            html = (
                "<span style='font-family:Courier New; "
                f"font-size:14px; color:{colour};'>"
                f"u {u_arr[i]:5.1f}  v {v_arr[i]:5.1f}  r {r_dps:7.3f}<br>"
                f"thr {thr_pct[i]:3.0f}% δ {rud_deg[i]:5.1f}°<br>"
                f"{roles[i]}"
                "</span>"
            )
            lbl = self.label_items[i]
            lbl.setHtml(html)
            # place label a tad off the bow so it never covers the hull
            lbl.setPos(x + 0.02 * L * np.cos(curr_hd),
                       y + 0.02 * L * np.sin(curr_hd))
            lbl.setZValue(100)
            #self.label_items.append(txt)

        # ── UPDATE ENVIRONMENT PANEL ───────────────────────────────────────
        self.lbl_time.setText(f"Time: {self.sim.t:.2f} s")
        # wind stats & vane
        wv = playful_wind(self.sim.state, self.sim.t)
        wx, wy = wv[:,0]
        ws = np.hypot(wx, wy)
        wd = np.degrees(np.arctan2(wy, wx)) % 360
        self.lbl_wind.setText(f"Wind: {ws:.2f} m/s, {wd:.0f}°")
        # ensure vane points even if speed zero
        self.wind_arrow.setRotation(-wd)
        # current stats & vane
        cv = north_south_current(self.sim.state, self.sim.t)
        cx, cy = cv[:,0]
        cs = np.hypot(cx, cy)
        cd = np.degrees(np.arctan2(cy, cx)) % 360
        self.lbl_current.setText(f"Current: {cs:.2f} m/s, {cd:.0f}°")
        self.current_arrow.setRotation(-cd)



