import sys
import numpy as np
import math
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
from PyQt5 import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem
from PyQt5.QtGui     import QPolygonF, QBrush, QPen, QColor, QTextCursor
from PyQt5.QtCore    import QPointF, QLineF
from pyqtgraph import RectROI, mkPen
from emergent.ship_abm.simulation_core import simulation, SIMULATION_BOUNDS
from emergent.ship_abm.simulation_core import playful_wind, north_south_current
from emergent.ship_abm.simulation_core import compute_zigzag_metrics
from datetime import datetime, date, timezone
import sqlite3
import json
import pyqtgraph.functions as fn
from pyproj import Transformer
import os
import json


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
        pts = [QPointF(float(x), float(y)) for x,y in body]
        self.setPolygon(QPolygonF(pts))
        self.setBrush(QBrush(QColor(*color)))
        pen = QPen(QColor(*pen_color))
        pen.setWidthF(pen_width)
        self.setPen(pen)


class ArrowField(pg.GraphicsObject):
    """Batched 2D arrow field renderer using a single QGraphicsObject.
    Stores shafts as QLineF and heads as QPolygonF and draws them in one paint call.
    """
    def __init__(self):
        super().__init__()
        self._lines = []      # list of QLineF
        self._heads = []      # list of QPolygonF
        self._pen = QPen(QColor(255,255,255))
        self._brush = QBrush(QColor(255,255,255))
        self._bounds = QtCore.QRectF(0,0,1,1)
        # instrumentation
        self._paint_count = 0
        self._set_count = 0
        self._last_paint_time = 0.0

    def setData(self, lines, heads, pen: QPen, brush: QBrush, z: float = 0):
        # lines: sequence of tuples (x0,y0,x1,y1)
        # heads: sequence of sequences of QPointF
        new_lines = [QLineF(a,b,c,d) for (a,b,c,d) in lines]
        new_heads = [QPolygonF(pts) for pts in heads]
        # compute new bounds
        xs = []
        ys = []
        for l in new_lines:
            xs += [l.x1(), l.x2()]
            ys += [l.y1(), l.y2()]
        for poly in new_heads:
            for p in poly:
                xs.append(p.x()); ys.append(p.y())
        if xs and ys:
            margin = max(2.0, pen.widthF() * 2)
            new_bounds = QtCore.QRectF(min(xs)-margin, min(ys)-margin,
                                       max(xs)-min(xs)+2*margin, max(ys)-min(ys)+2*margin)
        else:
            new_bounds = QtCore.QRectF(0,0,1,1)

        # Only call prepareGeometryChange if bounds actually changed to avoid scene relayout
        bounds_changed = (not hasattr(self, '_bounds')) or (new_bounds != self._bounds)
        if bounds_changed:
            try:
                self.prepareGeometryChange()
            except Exception:
                pass
            self._bounds = new_bounds

        # assign new geometry and styles
        self._lines = new_lines
        self._heads = new_heads
        self._pen = pen
        self._brush = brush
        # set Z value only if changed
        try:
            if not hasattr(self, '_zval') or self._zval != z:
                self.setZValue(z)
                self._zval = z
        except Exception:
            pass
        # request repaint (cheap)
        self.update()
        try:
            self._set_count += 1
            if self._set_count <= 5 or (self._set_count % 20) == 0:
                print(f"[ArrowField] setData called #{self._set_count} lines={len(self._lines)} heads={len(self._heads)} z={z}")
        except Exception:
            pass

    def boundingRect(self):
        return self._bounds

    def paint(self, painter, option, widget=None):
        if not self._lines and not self._heads:
            return
        painter.setPen(self._pen)
        # draw all shafts in one go
        import time
        t0 = time.time()
        try:
            painter.drawLines(self._lines)
        except Exception:
            # fallback per-line
            for ln in self._lines:
                painter.drawLine(ln)
        # draw heads
        painter.setBrush(self._brush)
        for poly in self._heads:
            painter.drawPolygon(poly)
        # instrumentation: occasionally report paint duration
        try:
            dt = time.time() - t0
            self._paint_count += 1
            if dt > 0.02 or (self._paint_count % 100) == 0:
                print(f"[ArrowField] paint #{self._paint_count} dur={dt:.4f}s lines={len(self._lines)} heads={len(self._heads)}")
            self._last_paint_time = dt
        except Exception:
            pass


class ship_viewer(QtWidgets.QWidget):
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
                 test_mode: str | None = None,
                 zigzag_deg: int = 10,
                 zigzag_hold: float = 40.0,
                 load_enc: bool = True,
                 sim_instance: object | None = None):
        super().__init__()
        self.setWindowTitle(f"Ship ABM Viewer – {port_name}")
        self.resize(1200, 800)
        self.ais_removed = False
        self.use_ais = use_ais
        self.ais_item = None
        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: black;")
        # Ensure core runtime attributes exist early to avoid races or
        # partial-initialization causing AttributeError in the timer loop.
        # These are lightweight placeholders and may be overridden later.
        try:
            if not hasattr(self, 'timer'):
                self.timer = None
        except Exception:
            self.timer = None
        # common item lists and placeholders
        self.ship_items = []
        # persistent UI flag marker items (one per ship) — small red ellipse
        self.flag_items = []
        self.route_item = None
        self.rudder_items = []
        self.heading_items = []
        self.label_items = []
        self.quiver_items = []
        self.wind_quiver_items = []
        # remember ENC catalog URL for deferred loading
        self._xml_url = xml_url

        # 1) Core simulation instantiation
        # If a preloaded simulation instance is provided, reuse it (it may contain enc_data)
        if sim_instance is not None:
            self.sim = sim_instance
            self._requested_enc_load = False
        else:
            # Construct simulation without loading ENC in the constructor to avoid blocking UI init.
            self._requested_enc_load = bool(load_enc)
            self.sim = simulation(
                port_name=port_name,
                dt=dt,
                T=T,
                n_agents=n_agents,
                coast_simplify_tol=coast_simplify_tol,
                light_bg=light_bg,
                verbose=verbose,
                use_ais=use_ais,
                load_enc=False,  # defer ENC loading until after UI is up
                test_mode=test_mode,
                zigzag_deg=zigzag_deg,
                zigzag_hold=zigzag_hold
            )

        # Helper: ensure the viewer window is raised on show/start
        def _bring_to_front():
            try:
                # Qt method: raise and activateWindow
                self.raise_()
                self.activateWindow()
                # Avoid toggling WindowStaysOnTopHint to prevent additional show()/flag changes
                # which can trigger extra paint events or OS-level reparenting causing strobe.
                try:
                    self.show()
                except Exception:
                    pass
            except Exception:
                pass

        self._bring_to_front = _bring_to_front

        # 2) Setup layouts
        layout = QtWidgets.QHBoxLayout(self)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.view = self.plot_widget.addViewBox()
        self.view.setAspectLocked(True)
        # no automatic quiver refresh on view changes; user will press Refresh
        layout.addWidget(self.plot_widget, stretch=4)

        # Create a scrollable control panel to avoid forcing the main window
        # to expand beyond the screen when many widgets are added during ENC draws.
        ctrl_container = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_container)
        ctrl_scroll = QtWidgets.QScrollArea()
        ctrl_scroll.setWidget(ctrl_container)
        ctrl_scroll.setWidgetResizable(True)
        # keep the control column a reasonable width
        ctrl_scroll.setMinimumWidth(300)
        ctrl_scroll.setMaximumWidth(520)
        layout.addWidget(ctrl_scroll, stretch=1)

        # (Removed) Simulation Log panel: kept sim.log_lines internally but
        # the on-screen text panel was removed to save vertical space.
        # By default we do not pipe simulation logs into the GUI to reduce
        # noisy updates and repaint churn. Set this True to enable GUI logs
        # if you re-add a log panel later.
        self.show_sim_logs = False

        # lightweight loading UI: hidden by default, shown while ENC/environment loads
        self.lbl_loading = QtWidgets.QLabel("")
        self.lbl_loading.setStyleSheet('color: black;')
        self.lbl_loading.setVisible(False)
        ctrl_layout.addWidget(self.lbl_loading)

        self.pb_loading = QtWidgets.QProgressBar()
        self.pb_loading.setRange(0, 0)  # indeterminate
        self.pb_loading.setVisible(False)
        ctrl_layout.addWidget(self.pb_loading)

        # helper wrapper: simulation core expects objects with set_text, set_fontsize, set_fontfamily
        class _LogTextArtist:
            def __init__(self, idx, panel, sim):
                # panel argument is ignored (no on-screen log panel)
                self.idx = idx
                self.sim = sim
            def set_text(self, txt):
                # keep sim.log_lines but avoid expensive GUI updates
                try:
                    lines = list(self.sim.log_lines)
                except Exception:
                    lines = []
                maxl = getattr(self.sim, 'max_log_lines', 10)
                self.sim.log_lines = lines[:maxl]
            def set_fontsize(self, *_):
                return
            def set_fontfamily(self, *_):
                return

        # create placeholder artists and attach to simulation so core can update them
        # Do not attach text artists by default (keep logs on console when verbose)
        try:
            self.sim.log_text_artists = None
        except Exception:
            self.sim.log_text_artists = None

        # helper to log (console + sim.log_lines). No on-screen panel is used
        def _ui_log(msg: str):
            try:
                if hasattr(self.sim, 'log_lines'):
                    self.sim.log_lines.insert(0, msg)
                # keep console output for diagnostics
                print(msg)
            except Exception:
                print(msg)

        # 3) Inset panel background
        # Temporarily disable widget updates while we build heavy basemap items
        try:
            self.setUpdatesEnabled(False)
        except Exception:
            pass

        self.panel_widget = pg.GraphicsLayoutWidget()
        self.panel_widget.setBackground(self.palette().color(self.backgroundRole()))
        try:
            # keep inset smaller so controls fit without scrolling
            self.panel_widget.setFixedSize(240, 240)
        except Exception:
            pass
        ctrl_layout.addWidget(self.panel_widget)
        self.inset = self.panel_widget.addViewBox()
        self.inset.setAspectLocked(True)
        self.inset.setMouseEnabled(False, False)
        
        # create a fixed-size RectROI and add it to the inset viewbox
        self.extent_roi = RectROI([0, 0], [0, 0],
                                  pen=mkPen('r', width=2),
                                  movable=False)
        self.extent_roi.setZValue(100)            # sit on top of the map
        self.inset.addItem(self.extent_roi)
        
        # draw full ENC basemap (land & shoreline) into inset
        if self.sim.enc_data:
            # 1) land polygons (LNDARE), fill tan
            if 'LNDARE' in self.sim.enc_data:
                for geom in self.sim.enc_data['LNDARE'].geometry:
                    # handle both Polygons and LineStrings
                    if hasattr(geom, "exterior"):
                        coords = np.array(geom.exterior.coords)
                    else:
                        coords = np.array(geom.coords)

                    land_item = ship_polygon(coords,
                                             color=(200,180,100,200),
                                             pen_color=(150,120,80))
                    self.inset.addItem(land_item)
            # 2) shoreline (COALNE), draw thin gray
            # DISABLED: Too many coastline segments cause pyqtgraph to hang during bounds calculation
            # TODO: Merge geometries or simplify before drawing
            # if 'COALNE' in self.sim.enc_data:
            #     for geom in self.sim.enc_data['COALNE'].geometry:
            #         # handle both Polygons and LineStrings
            #         if hasattr(geom, "exterior"):
            #             coords = np.array(geom.exterior.coords)
            #         else:
            #             coords = np.array(geom.coords)
            #         
            #         shore_item = ship_polygon(coords,
            #                                   color=(0,0,0,0),
            #                                   pen_color=(100,100,100))
            #         self.inset.addItem(shore_item)
                    
        
        # 4) Draw basemap
        if self.sim.enc_data:
            self._draw_basemap()
        else:
            # Fallback: draw a simple water rectangle for quick UI runs when ENC is disabled
            try:
                xmin, xmax = self.sim.minx, self.sim.maxx
                ymin, ymax = self.sim.miny, self.sim.maxy
                w = xmax - xmin
                h = ymax - ymin
                water = QGraphicsRectItem(xmin, ymin, w, h)
                water.setBrush(QBrush(QColor(200, 230, 255)))
                water.setPen(QPen(QColor(200, 230, 255)))
                water.setZValue(-10)
                self.view.addItem(water)
                self._fallback_water = water
            except Exception:
                self._fallback_water = None

        # re-enable updates and process events so the UI paints once
        try:
            self.setUpdatesEnabled(True)
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        # Initialize quiver items as empty - will be populated when env loads
        self.quiver_items = []
        self.wind_quiver_items = []

        # Date/time picker for environmental forcing (day+time)
        self.lbl_datetime = QtWidgets.QLabel("Forcing date/time (UTC):")
        self.dt_edit = QtWidgets.QDateTimeEdit()
        self.dt_edit.setCalendarPopup(True)
        self.dt_edit.setDateTime(datetime.utcnow())
        self.dt_edit.setDisplayFormat('yyyy-MM-dd HH:mm')
        ctrl_layout.addWidget(self.lbl_datetime)
        ctrl_layout.addWidget(self.dt_edit)

        # Events panel (collision / allision summaries)
        ctrl_layout.addWidget(QtWidgets.QLabel("Events:"))
        self.events_view = QtWidgets.QTextEdit()
        self.events_view.setReadOnly(True)
        try:
            self.events_view.setMaximumHeight(150)
        except Exception:
            pass
        ctrl_layout.addWidget(self.events_view)
        self.clear_events_btn = QtWidgets.QPushButton("Clear Events")
        ctrl_layout.addWidget(self.clear_events_btn)
        try:
            self.clear_events_btn.clicked.connect(self._clear_events)
        except Exception:
            pass
        # wire up later once self is fully initialized

        # persisted routes file paths (loading deferred until after controls exist)
        self._routes_path = os.path.join(os.path.expanduser('~'), '.emergent_routes.json')
        self._named_routes_path = os.path.join(os.path.expanduser('~'), '.emergent_named_routes.json')

        # Route manager UI (named routes)
        try:
            ctrl_layout.addWidget(QtWidgets.QLabel("Saved routes:"))
            self.lst_routes = QtWidgets.QListWidget()
            try:
                self.lst_routes.setMaximumHeight(120)
            except Exception:
                pass
            ctrl_layout.addWidget(self.lst_routes)
            rbtn_row = QtWidgets.QHBoxLayout()
            self.btn_save_route_as = QtWidgets.QPushButton("Save Route As...")
            self.btn_load_route = QtWidgets.QPushButton("Load Selected")
            self.btn_delete_route = QtWidgets.QPushButton("Delete Selected")
            rbtn_row.addWidget(self.btn_save_route_as)
            rbtn_row.addWidget(self.btn_load_route)
            rbtn_row.addWidget(self.btn_delete_route)
            # button to explicitly load the last persisted (autosaved) route
            self.btn_load_persisted = QtWidgets.QPushButton("Load Persisted")
            rbtn_row.addWidget(self.btn_load_persisted)

            ctrl_layout.addLayout(rbtn_row)
            self.btn_save_route_as.clicked.connect(self._save_route_as_dialog)
            self.btn_load_route.clicked.connect(self._load_selected_named_route)
            self.btn_delete_route.clicked.connect(self._delete_selected_named_route)
            try:
                self.btn_load_persisted.clicked.connect(self._apply_persisted_route)
            except Exception:
                pass
            # populate list
            try:
                if os.path.exists(self._named_routes_path):
                    with open(self._named_routes_path, 'r') as fh:
                        data = json.load(fh)
                        for name in data.get(self.sim.port_name, {}):
                            self.lst_routes.addItem(name)
            except Exception:
                pass
        except Exception:
            pass
        
        # Environmental forcing will be loaded when user clicks "Start Simulation"
        # Environmental controls are set up in _init_gui_controls
        self._init_gui_controls(test_mode, ctrl_layout)

        # Now that GUI controls exist, detect any persisted routes but DO NOT auto-apply them.
        # The user must explicitly click 'Load Persisted' to apply a saved route to the simulation.
        # NOTE: intentionally do NOT load or detect persisted routes at startup.
        # Users must click the 'Load Persisted' button to read and apply saved routes.

        # Environmental forcing will be loaded when user clicks "Start Simulation"
        # to avoid Qt threading issues during init

    def _load_environmental_and_quivers(self):
        """
        Load environmental forcing data and draw quiver plots.
        Called via QTimer after Qt event loop is running to avoid threading issues.
        """
        # Show loading UI
        try:
            self.lbl_loading.setText("Loading environmental forcing (ENCs/winds/currents)...")
            self.lbl_loading.setVisible(True)
            self.pb_loading.setVisible(True)
        except Exception:
            pass

        # Poll sim.log_lines into the log panel so ENC fetch/extract messages are visible
        def _poll_logs():
            try:
                lines = list(getattr(self.sim, 'log_lines', []))
                maxl = getattr(self.sim, 'max_log_lines', 200)
                lines = lines[:maxl]
                # Always print logs to console when verbose is enabled on sim
                if getattr(self.sim, 'verbose', False):
                    for ln in lines:
                        try:
                            print(f"[SimLog] {ln}")
                        except Exception:
                            pass
                # Update GUI when user requests it (or during a forced reload where the checkbox is set)
                if getattr(self, 'show_sim_logs', False) or getattr(self, 'force_stream_logs', False) or getattr(self, 'chk_stream_logs', None) and self.chk_stream_logs.isChecked():
                    out = []
                    for ln in lines:
                        try:
                            ts = datetime.now().strftime('%H:%M:%S')
                            out.append(f"[{ts}] {ln}")
                        except Exception:
                            out.append(ln)
                    self.log_panel.setPlainText("\n".join(out))
                    try:
                        self.log_panel.moveCursor(QTextCursor.End)
                    except Exception:
                        pass
            except Exception:
                pass

        self._env_log_timer = QtCore.QTimer(self)
        self._env_log_timer.setInterval(500)
        self._env_log_timer.timeout.connect(_poll_logs)
        self._env_log_timer.start()

        class _EnvLoaderThread(QtCore.QThread):
            def __init__(self, sim):
                super().__init__()
                self.sim = sim
                self._error = None
            def run(self):
                try:
                    print("[ShipViewer] Loading environmental forcing (thread)...")
                    self.sim.load_environmental_forcing()
                except Exception as e:
                    self._error = str(e)

        try:
            # if user asked to stream logs during reloads, enable temporary flag
            prev_stream = getattr(self, 'force_stream_logs', False)
            if getattr(self, 'chk_stream_logs', None) and self.chk_stream_logs.isChecked():
                self.force_stream_logs = True

            self._env_thread = _EnvLoaderThread(self.sim)
            def _on_env_done():
                try:
                    self._env_log_timer.stop()
                except Exception:
                    pass
                try:
                    self.lbl_loading.setVisible(False)
                    self.pb_loading.setVisible(False)
                except Exception:
                    pass
                # ensure UI controls are restored after heavy IO
                try:
                    self._restore_control_visibility()
                except Exception:
                    pass
                # restore streaming flag
                try:
                    if getattr(self, 'force_stream_logs', False) and not getattr(self, 'chk_stream_logs', None) or not (getattr(self, 'chk_stream_logs', None) and self.chk_stream_logs.isChecked()):
                        # clear the temporary flag if user didn't request persistent streaming
                        self.force_stream_logs = False
                except Exception:
                    pass
                try:
                    if getattr(self._env_thread, '_error', None):
                        print(f"[ShipViewer] Environmental load failed: {self._env_thread._error}")
                except Exception:
                    pass
                try:
                    print("[ShipViewer] Drawing current and wind quivers...")
                    self._draw_quivers()
                except Exception as e:
                    print(f"[ShipViewer] Failed to draw quivers after load: {e}")
                # Re-enable Start button when environmental forcing is loaded
                try:
                    if getattr(self.sim, '_env_loaded', False):
                        try:
                            self.btn_start_sim.setEnabled(True)
                        except Exception:
                            pass
                except Exception:
                    pass

            self._env_thread.finished.connect(_on_env_done)
            self._env_thread.start()
        except Exception as e:
            print(f"[ShipViewer] Thread failed to start, loading synchronously: {e}")
            try:
                self.sim.load_environmental_forcing()
                _poll_logs()
                self._draw_quivers()
            except Exception as e2:
                print(f"[ShipViewer] Sync load failed: {e2}")
    
    def _init_gui_controls(self, test_mode, ctrl_layout):
        """Initialize GUI controls - called from __init__ after quiver setup is deferred."""
        # ── status labels ────────────────────────────────────────────
        self.lbl_time    = QtWidgets.QLabel("Time: 0.00 s")
        self.lbl_wind    = QtWidgets.QLabel("Wind: 0.00 m/s, 0°")
        self.lbl_current = QtWidgets.QLabel("Current: 0.00 m/s, 0°")
        for lbl in (self.lbl_time, self.lbl_wind, self.lbl_current):
            lbl.setStyleSheet("color: black;")
            ctrl_layout.addWidget(lbl)

        # 6) Define Route button + first Start Simulation only
        if test_mode != "zigzag":
            self.btn_define_route = QtWidgets.QPushButton("Define Route")
            self.btn_define_route.clicked.connect(self._start_route_mode)
            ctrl_layout.addWidget(self.btn_define_route)
            self._status_label.setText(f"Draw route for agent 1 of {self.sim.n}")
            instr = QtWidgets.QLabel(
                 "> Define Route:\n"
                "   • Left-click on map to add points\n"
                "   • Right-click to finish (needs ≥2 points)\n"
                 "\n> Start Simulation when ready"
            )
            instr.setWordWrap(True)
            instr.setStyleSheet("color: black;")
            ctrl_layout.addWidget(instr)
            self._route_pts_by_agent = [[] for _ in range(self.sim.n)]
            self._current_route_agent = 0
            self._route_pts = []  # temporary holder for current agent's clicks
        
        # start button
        self.btn_start_sim = QtWidgets.QPushButton("Start Simulation")
        self.btn_start_sim.clicked.connect(self._start_simulation)
        # If ENC or environmental forcing is requested (or not yet loaded), disable Start until ready
        try:
            if getattr(self, '_requested_enc_load', False) or (not getattr(self.sim, '_env_loaded', False)):
                self.btn_start_sim.setEnabled(False)
        except Exception:
            pass
        ctrl_layout.addWidget(self.btn_start_sim)
        
        # kill switch
        self.btn_kill_power = QtWidgets.QPushButton("Kill Power")
        self.btn_kill_power.clicked.connect(self._on_kill_power)
        ctrl_layout.addWidget(self.btn_kill_power)

        # Pause / Play controls
        play_row = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play)
        # initially, play is disabled until simulation runs
        self.btn_play.setEnabled(False)
        play_row.addWidget(self.btn_pause)
        play_row.addWidget(self.btn_play)
        ctrl_layout.addLayout(play_row)

        # ── Quiver controls (density/scale/pen width) ─────────────────
        qlabel = QtWidgets.QLabel("Quiver density (step, 1=full):")
        self.spin_quiver_step = QtWidgets.QSpinBox()
        self.spin_quiver_step.setRange(1, 16)
        # default to full density
        self.spin_quiver_step.setValue(getattr(self, 'quiver_step', 1))
        self.spin_quiver_step.valueChanged.connect(lambda v: (setattr(self, 'quiver_step', v), self._save_user_config()))
        ctrl_layout.addWidget(qlabel)
        ctrl_layout.addWidget(self.spin_quiver_step)

        qlabel2 = QtWidgets.QLabel("Quiver scale (fraction of grid spacing):")
        self.dspin_quiver_scale = QtWidgets.QDoubleSpinBox()
        # keep base scale smaller so arrows are not oversized by default
        self.dspin_quiver_scale.setRange(0.05, 2.0)
        self.dspin_quiver_scale.setSingleStep(0.05)
        self.dspin_quiver_scale.setValue(getattr(self, 'quiver_scale', 0.35))
        self.dspin_quiver_scale.valueChanged.connect(lambda v: (setattr(self, 'quiver_scale', v), self._save_user_config()))
        ctrl_layout.addWidget(qlabel2)
        ctrl_layout.addWidget(self.dspin_quiver_scale)

        qlabel3 = QtWidgets.QLabel("Quiver pen width:")
        self.spin_quiver_pen = QtWidgets.QSpinBox()
        self.spin_quiver_pen.setRange(1, 8)
        self.spin_quiver_pen.setValue(getattr(self, 'quiver_pen_width', 2))
        self.spin_quiver_pen.valueChanged.connect(lambda v: (setattr(self, 'quiver_pen_width', v), self._save_user_config()))
        ctrl_layout.addWidget(qlabel3)
        ctrl_layout.addWidget(self.spin_quiver_pen)

        # Mask on-land quivers (toggle)
        self.chk_mask_land = QtWidgets.QCheckBox("Mask quivers on land")
        self.chk_mask_land.setChecked(True)
        ctrl_layout.addWidget(self.chk_mask_land)
        self.chk_smooth_quivers = QtWidgets.QCheckBox("Smooth quivers (display only)")
        self.chk_smooth_quivers.setChecked(False)
        ctrl_layout.addWidget(self.chk_smooth_quivers)

    # Auto-clamp toggle: when enabled, the viewer will increase quiver step
        self.chk_auto_clamp = QtWidgets.QCheckBox("Auto-clamp quiver density")
        self.chk_auto_clamp.setChecked(True)
        ctrl_layout.addWidget(self.chk_auto_clamp)

        # Refresh quivers now button
        self.btn_refresh_quivers = QtWidgets.QPushButton("Refresh Quivers Now")
        self.btn_refresh_quivers.clicked.connect(self._refresh_quivers)
        ctrl_layout.addWidget(self.btn_refresh_quivers)

        # Zoom controls: allow quick zoom in/out and center on primary ship
        zrow = QtWidgets.QHBoxLayout()
        self.btn_zoom_in = QtWidgets.QPushButton("Zoom In")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(0.7))
        self.btn_zoom_out = QtWidgets.QPushButton("Zoom Out")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(1.4))
        self.btn_center_ship = QtWidgets.QPushButton("Center on Ship")
        self.btn_center_ship.clicked.connect(self._center_on_ship)
        # add Zoom In first (was previously created but not added, causing it to float)
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_center_ship)
        ctrl_layout.addLayout(zrow)

        # Legend for quivers
        legend_html = "<div><span style='color:rgb(0,100,255);font-weight:bold;'>▮</span> Currents &nbsp;&nbsp; " \
                  "<span style='color:rgb(160,32,240);font-weight:bold;'>▮</span> Winds</div>"
        lbl_legend = QtWidgets.QLabel()
        lbl_legend.setText(legend_html)
        lbl_legend.setStyleSheet('font-size:12px;')
        ctrl_layout.addWidget(lbl_legend)

        # Load user config (quiver settings)
        self._config_path = os.path.join(os.path.expanduser('~'), '.emergent_config.json')
        self._load_user_config()

        # settings are saved by the individual widget handlers above

        # iterations & production
        self.lbl_iterations = QtWidgets.QLabel("Iterations:")
        self.spin_iterations = QtWidgets.QSpinBox()
        self.spin_iterations.setRange(1, 1000)
        self.btn_prod_run = QtWidgets.QPushButton("Run Production")
        ctrl_layout.addWidget(self.lbl_iterations)
        ctrl_layout.addWidget(self.spin_iterations)
        ctrl_layout.addWidget(self.btn_prod_run)
        self.btn_prod_run.clicked.connect(self._run_production)

        # Compact ENC/quiver/env control row (save vertical space)
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        self.btn_quiver = QtWidgets.QPushButton("Quiver")
        self.btn_quiver.setFixedHeight(24)
        self.btn_quiver.setToolTip("Clear ENC cache & redownload (compact)")
        self.btn_quiver.clicked.connect(self._on_clear_enc_cache)
        row_layout.addWidget(self.btn_quiver)

        self.btn_quiver_alt = QtWidgets.QPushButton("Quiver ?")
        self.btn_quiver_alt.setFixedHeight(24)
        self.btn_quiver_alt.setToolTip("Select ENC cell to clear (compact)")
        self.btn_quiver_alt.clicked.connect(self._on_selective_clear)
        row_layout.addWidget(self.btn_quiver_alt)

        self.btn_load_env = QtWidgets.QPushButton("Load Env")
        self.btn_load_env.setFixedHeight(24)
        self.btn_load_env.setToolTip("Retry environmental forcings load now")
        self.btn_load_env.clicked.connect(lambda: self._load_environmental_and_quivers())
        row_layout.addWidget(self.btn_load_env)

        # explicit basemap refresh for quick recovery when coastlines don't appear
        self.btn_refresh_basemap = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_basemap.setFixedHeight(24)
        self.btn_refresh_basemap.setToolTip("Redraw basemap / coastlines")
        def _on_refresh_basemap():
            try:
                self._draw_basemap()
                print("[ShipViewer] Manual basemap refresh called")
            except Exception as e:
                print(f"[ShipViewer] Manual basemap refresh failed: {e}")
        self.btn_refresh_basemap.clicked.connect(_on_refresh_basemap)
        row_layout.addWidget(self.btn_refresh_basemap)
        # (Debug Env button removed to reduce control pane width)

        ctrl_layout.addWidget(row)

        # show
        self.show()

    # ----- ENC progress poller helpers ---------------------------------
    def _start_enc_progress_poller(self):
        """Begin polling self.sim._enc_progress and update pb_loading"""
        try:
            # make determinate
            self.pb_loading.setRange(0, 100)
            # ensure label visible
            self.lbl_loading.setVisible(True)
            # create timer if missing
            if getattr(self, '_enc_progress_timer', None) is None:
                self._enc_progress_timer = QtCore.QTimer(self)
                self._enc_progress_timer.setInterval(400)
            # connect safely (avoid duplicate connections)
            try:
                self._enc_progress_timer.timeout.disconnect()
            except Exception:
                pass
            def _poll_enc_progress():
                try:
                    prog = getattr(self.sim, '_enc_progress', None)
                    if prog is None:
                        return
                    if isinstance(prog, float) and 0.0 <= prog <= 1.0:
                        val = int(prog * 100)
                    else:
                        try:
                            val = int(prog)
                        except Exception:
                            return
                    self.pb_loading.setValue(val)
                    self.lbl_loading.setText(f"Loading ENCs... {val}%")
                    if val >= 100:
                        try:
                            self._enc_progress_timer.stop()
                        except Exception:
                            pass
                except Exception:
                    pass
            self._enc_progress_timer.timeout.connect(_poll_enc_progress)
            self._enc_progress_timer.start()
        except Exception:
            pass

    def _restore_control_visibility(self):
        """Ensure key control widgets are visible/enabled after heavy operations."""
        try:
            keys = [
                'btn_zoom_in', 'btn_zoom_out', 'btn_center_ship',
                'btn_refresh_quivers', 'btn_define_route', 'btn_start_sim',
                'btn_kill_power', 'btn_quiver', 'btn_quiver_alt', 'btn_load_env', 'btn_refresh_basemap'
            ]
            for k in keys:
                try:
                    w = getattr(self, k, None)
                    if w is None:
                        continue
                    try:
                        w.setVisible(True)
                    except Exception:
                        pass
                    try:
                        w.setEnabled(True)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _debug_control_layout(self):
        """Print debug information about key control widgets to the console."""
        try:
            keys = [
                'btn_zoom_in', 'btn_zoom_out', 'btn_center_ship',
                'btn_refresh_quivers', 'btn_define_route', 'btn_start_sim',
                'btn_kill_power', 'btn_quiver', 'btn_quiver_alt', 'btn_load_env',
                'lst_routes', 'btn_save_route_as'
            ]
            print("[ShipViewer DEBUG] Control layout snapshot:")
            for k in keys:
                try:
                    w = getattr(self, k, None)
                    if w is None:
                        print(f"  {k}: MISSING")
                        continue
                    sh = w.sizeHint() if hasattr(w, 'sizeHint') else None
                    mn = w.minimumSize() if hasattr(w, 'minimumSize') else None
                    mx = w.maximumSize() if hasattr(w, 'maximumSize') else None
                    print(f"  {k}: visible={w.isVisible()} enabled={w.isEnabled()} sizeHint={sh} min={mn} max={mx}")
                except Exception as e:
                    print(f"  {k}: ERROR {e}")
        except Exception:
            pass

    def _stop_enc_progress_poller(self):
        try:
            if getattr(self, '_enc_progress_timer', None):
                try:
                    self._enc_progress_timer.stop()
                except Exception:
                    pass
            # restore indeterminate default for other loads
            try:
                self.pb_loading.setRange(0, 0)
                self.pb_loading.setValue(0)
            except Exception:
                pass
            try:
                self.lbl_loading.setVisible(False)
            except Exception:
                pass
        except Exception:
            pass

        # If ENC was requested, load it in a background thread now that the UI is visible
        try:
            if getattr(self, '_requested_enc_load', False):
                class _ENCThread(QtCore.QThread):
                    def __init__(self, sim, xml_url, verbose=False):
                        super().__init__()
                        self.sim = sim
                        self.xml_url = xml_url
                        self.verbose = verbose
                        self._error = None
                    def run(self):
                        try:
                            print("[ShipViewer] Background ENC load starting...")
                            self.sim.load_enc_features(self.xml_url, verbose=self.verbose)
                        except Exception as e:
                            self._error = str(e)

                self._enc_thread = _ENCThread(self.sim, self._xml_url, verbose=getattr(self, 'verbose', False))
                # start ENC progress poller when ENC thread starts
                def _start_enc_progress_poller():
                    try:
                        # switch progress bar to determinate when enc publishes progress
                        self.pb_loading.setRange(0, 100)
                        # create a timer to poll sim._enc_progress
                        self._enc_progress_timer = QtCore.QTimer(self)
                        self._enc_progress_timer.setInterval(400)
                        def _poll_enc_progress():
                            try:
                                prog = getattr(self.sim, '_enc_progress', None)
                                if prog is None:
                                    # unknown progress: stay indeterminate
                                    return
                                # prog expected 0.0-1.0 or 0-100
                                if isinstance(prog, float) and 0.0 <= prog <= 1.0:
                                    val = int(prog * 100)
                                else:
                                    try:
                                        val = int(prog)
                                    except Exception:
                                        return
                                self.pb_loading.setValue(val)
                                self.lbl_loading.setText(f"Loading ENCs... {val}%")
                                # if complete or 100%, stop
                                if val >= 100:
                                    try:
                                        self._enc_progress_timer.stop()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        self._enc_progress_timer.timeout.connect(_poll_enc_progress)
                        self._enc_progress_timer.start()
                    except Exception:
                        pass

                # start poller just before thread runs
                _start_enc_progress_poller()
                def _on_enc_done():
                    try:
                        if getattr(self._enc_thread, '_error', None):
                            print(f"[ShipViewer] ENC load failed: {self._enc_thread._error}")
                    except Exception:
                        pass
                    # draw basemap and refresh inset
                    try:
                        if getattr(self.sim, 'enc_data', None):
                            # ensure coastline exists; synthesize from LNDARE if missing
                            try:
                                synthesized = False
                                if hasattr(self.sim, 'synthesize_coastline'):
                                    synthesized = self.sim.synthesize_coastline()
                                    if synthesized:
                                        print("[ShipViewer] synthesize_coastline() created COALNE")
                            except Exception as e:
                                print(f"[ShipViewer] synthesize_coastline error: {e}")

                            # diagnostic: print which ENC layers are present vs missing
                            try:
                                expected = ['LNDARE','COALNE','DEPARE','DEPVAL','BRIDGE','BOY','DRVAL2','OBSTRN']
                                present = [k for k,v in (self.sim.enc_data or {}).items() if v is not None and (not getattr(v,'empty', False))]
                                missing = [k for k in expected if k not in present]
                                print(f"[ShipViewer] ENC layers present: {present}")
                                if missing:
                                    print(f"[ShipViewer] ENC layers missing: {missing}")
                            except Exception:
                                pass

                            # attempt to draw basemap and force a UI paint
                            try:
                                self._draw_basemap()
                                QtWidgets.QApplication.processEvents()
                                print("[ShipViewer] _draw_basemap() called after ENC load")
                            except Exception as e:
                                print(f"[ShipViewer] _draw_basemap() raised: {e}")
                            # diagnostic: report which ENC layers are present vs missing
                        # re-enable Start if ENC loaded (environment may still need to load winds/currents)
                        try:
                            # If env not loaded, start it with selected date so Start enables automatically
                            if not getattr(self.sim, '_env_loaded', False):
                                # use selected date/time from dt_edit
                                dt_sel = self.dt_edit.dateTime().toPyDateTime()
                                start_dt = dt_sel

                                # spawn env loader thread
                                class _EnvLoaderThread3(QtCore.QThread):
                                    def __init__(self, sim, start_dt):
                                        super().__init__()
                                        self.sim = sim
                                        self.start_dt = start_dt
                                        self._error = None
                                    def run(self):
                                        try:
                                            self.sim.load_environmental_forcing(start=self.start_dt)
                                        except Exception as e:
                                            self._error = str(e)

                                self._env_thread = _EnvLoaderThread3(self.sim, start_dt)

                                def _on_env_done_after_enc_load():
                                    try:
                                        if getattr(self._env_thread, '_error', None):
                                            print(f"[ShipViewer] Environmental load failed: {self._env_thread._error}")
                                    except Exception:
                                        pass
                                    try:
                                        if getattr(self.sim, '_env_loaded', False) or getattr(self.sim, 'enc_data', None):
                                            try:
                                                self.btn_start_sim.setEnabled(True)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                self._env_thread.finished.connect(_on_env_done_after_enc_load)
                                self._env_thread.start()
                            else:
                                try:
                                    if getattr(self.sim, '_env_loaded', False) or getattr(self.sim, 'enc_data', None):
                                        try:
                                            self.btn_start_sim.setEnabled(True)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[ShipViewer] Failed to draw basemap after ENC load: {e}")

                self._enc_thread.finished.connect(_on_enc_done)
                self._enc_thread.start()
        except Exception:
            pass

    def _on_clear_enc_cache(self):
        """
        Handler for 'Clear ENC Cache & Redownload' button.
        Deletes the local ENC cache directory and restarts the ENC loader thread.
        """
        # diagnostic snapshot: print current control states and proactively restore
        try:
            try:
                self._debug_control_layout()
            except Exception:
                pass
            try:
                self._restore_control_visibility()
            except Exception:
                pass
        except Exception:
            pass

        try:
            cache_root = os.path.join(os.path.expanduser('~'), '.emergent_cache', 'enc')
            if os.path.exists(cache_root):
                print(f"[ShipViewer] Removing ENC cache at {cache_root} ...")
                # remove tree carefully
                import shutil
                try:
                    shutil.rmtree(cache_root)
                except Exception as e:
                    print(f"[ShipViewer] Failed to remove cache dir: {e}")
            else:
                print(f"[ShipViewer] ENC cache not present at {cache_root}")
        except Exception as e:
            print(f"[ShipViewer] Error while clearing ENC cache: {e}")

        # Start background ENC reload (use same pattern as initial load)
        try:
            class _ENCReloadThread(QtCore.QThread):
                def __init__(self, sim, xml_url, verbose=False):
                    super().__init__()
                    self.sim = sim
                    self.xml_url = xml_url
                    self.verbose = verbose
                    self._error = None
                def run(self):
                    try:
                        print("[ShipViewer] ENC reload starting (thread)...")
                        self.sim.load_enc_features(self.xml_url, verbose=self.verbose)
                    except Exception as e:
                        self._error = str(e)

            self.lbl_loading.setText("Clearing cache and re-downloading ENCs...")
            self.lbl_loading.setVisible(True)
            self.pb_loading.setVisible(True)

            # debug snapshot after showing loading UI
            try:
                self._debug_control_layout()
            except Exception:
                pass

            # ensure determinate progress polling starts for reloads as well
            try:
                self._start_enc_progress_poller()
            except Exception:
                pass

            self._enc_thread = _ENCReloadThread(self.sim, self._xml_url, verbose=getattr(self, 'verbose', False))

            def _on_enc_reload_done():
                try:
                    if getattr(self._enc_thread, '_error', None):
                        print(f"[ShipViewer] ENC reload failed: {self._enc_thread._error}")
                except Exception:
                    pass
                # draw basemap and refresh inset
                try:
                    if getattr(self.sim, 'enc_data', None):
                        self._draw_basemap()
                    # If environmental forcing isn't loaded yet, start it now so Start can enable automatically
                    try:
                        if not getattr(self.sim, '_env_loaded', False):
                            # reuse Env loader thread pattern
                            class _EnvLoaderThread2(QtCore.QThread):
                                def __init__(self, sim):
                                    super().__init__()
                                    self.sim = sim
                                    self._error = None
                                def run(self):
                                    try:
                                        print('[ShipViewer] Loading environmental forcing (post-ENC reload)...')
                                        self.sim.load_environmental_forcing()
                                    except Exception as e:
                                        self._error = str(e)

                            self._env_thread = _EnvLoaderThread2(self.sim)

                            def _on_env_done_after_reload():
                                try:
                                    if getattr(self._env_thread, '_error', None):
                                        print(f"[ShipViewer] Environmental load failed after ENC reload: {self._env_thread._error}")
                                except Exception:
                                    pass
                                # Attempt to enable Start if env or enc present
                                try:
                                    if getattr(self.sim, '_env_loaded', False) or getattr(self.sim, 'enc_data', None):
                                        try:
                                            self.btn_start_sim.setEnabled(True)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass

                            self._env_thread.finished.connect(_on_env_done_after_reload)
                            self._env_thread.start()
                        else:
                            try:
                                if getattr(self.sim, '_env_loaded', False) or getattr(self.sim, 'enc_data', None):
                                    try:
                                        self.btn_start_sim.setEnabled(True)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        # restore control visibility in case UI elements were hidden
                        try:
                            self._restore_control_visibility()
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[ShipViewer] Failed to draw basemap after ENC reload: {e}")
                try:
                    self.lbl_loading.setVisible(False)
                    self.pb_loading.setVisible(False)
                except Exception:
                    pass

            self._enc_thread.finished.connect(_on_enc_reload_done)
            self._enc_thread.start()

            # debug snapshot immediately after thread start
            try:
                self._debug_control_layout()
            except Exception:
                pass
        except Exception as e:
            print(f"[ShipViewer] Failed to start ENC reload thread: {e}")

    def _on_selective_clear(self):
        """Open a directory picker rooted at the ENC cache so user can delete a single cell folder."""
        try:
            cache_root = os.path.join(os.path.expanduser('~'), '.emergent_cache', 'enc')
            if not os.path.exists(cache_root):
                QtWidgets.QMessageBox.information(self, "No ENC Cache", f"No ENC cache found at {cache_root}")
                return
            # ask user to pick a subdirectory
            dlg = QtWidgets.QFileDialog(self, "Select ENC cell folder to remove")
            dlg.setFileMode(QtWidgets.QFileDialog.Directory)
            dlg.setDirectory(cache_root)
            if dlg.exec_():
                sel = dlg.selectedFiles()
                if sel:
                    path = sel[0]
                    import shutil
                    try:
                        shutil.rmtree(path)
                        QtWidgets.QMessageBox.information(self, "Removed", f"Removed {path}")
                        print(f"[ShipViewer] Removed ENC cell cache {path}")
                    except Exception as e:
                        QtWidgets.QMessageBox.warning(self, "Remove failed", str(e))
        except Exception as e:
            print(f"[ShipViewer] Selective clear failed: {e}")

        # Create compact vane graphics in inset for wind and current
        try:
            # small triangular arrow for wind (purple)
            wind_poly = QPolygonF([QPointF(0, -8), QPointF(4, 4), QPointF(-4, 4)])
            self.wind_arrow = QGraphicsPolygonItem(wind_poly)
            self.wind_arrow.setBrush(QBrush(QColor(160,32,240)))
            self.wind_arrow.setPen(QPen(QColor(80,10,120)))
            self.wind_arrow.setZValue(200)
            # rotate around the triangle centroid (0,0) so the vane pivots naturally
            try:
                self.wind_arrow.setTransformOriginPoint(0, 0)
            except Exception:
                pass
            self.wind_arrow.setPos(40, 20)
            self.inset.addItem(self.wind_arrow)

            # small triangular arrow for current (blue)
            cur_poly = QPolygonF([QPointF(0, -6), QPointF(3, 3), QPointF(-3, 3)])
            self.current_arrow = QGraphicsPolygonItem(cur_poly)
            self.current_arrow.setBrush(QBrush(QColor(0,100,255)))
            self.current_arrow.setPen(QPen(QColor(0,60,160)))
            self.current_arrow.setZValue(200)
            # rotate around the triangle centroid (0,0)
            try:
                self.current_arrow.setTransformOriginPoint(0, 0)
            except Exception:
                pass
            self.current_arrow.setPos(40, 40)
            self.inset.addItem(self.current_arrow)
        except Exception:
            self.wind_arrow = None
            self.current_arrow = None
        
        # ─────────────────────────────────────────────────────────────────────
        # ANIMATION TIMER: drives _update_frame on each dt tick
        # ─────────────────────────────────────────────────────────────────────
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        # quiver refresh remains manual via the Refresh button

        # placeholders
        self.ship_items = []
        self.route_item = None
        self.rudder_items = []
        self.heading_items = []
        self.label_items = []
        # extra visuals: bow->waypoint lines and danger cones
        self.bow_goal_items = []
        self.danger_cone_items = []
        # trajectory line items (updated each tick)
        self.traj_items = []

    def _load_user_config(self):
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as fh:
                    cfg = json.load(fh)
                    self.quiver_step = int(cfg.get('quiver_step', getattr(self, 'quiver_step', 2)))
                    self.quiver_scale = float(cfg.get('quiver_scale', getattr(self, 'quiver_scale', 0.6)))
                    self.quiver_pen_width = int(cfg.get('quiver_pen_width', getattr(self, 'quiver_pen_width', 3)))
                    # set widget values if present
                    try: self.spin_quiver_step.setValue(self.quiver_step)
                    except Exception: pass
                    try: self.dspin_quiver_scale.setValue(self.quiver_scale)
                    except Exception: pass
                    try: self.spin_quiver_pen.setValue(self.quiver_pen_width)
                    except Exception: pass
        except Exception:
            pass

    def _save_user_config(self):
        try:
            cfg = {
                'quiver_step': getattr(self, 'quiver_step', 2),
                'quiver_scale': getattr(self, 'quiver_scale', 0.6),
                'quiver_pen_width': getattr(self, 'quiver_pen_width', 3)
            }
            with open(self._config_path, 'w') as fh:
                json.dump(cfg, fh)
        except Exception:
            pass

    def _refresh_quivers(self):
        # Force immediate quiver redraw by calling the quiver drawing helper
        try:
            # remove existing quiver items and trigger the draw
            for itm in list(self.quiver_items) + list(self.wind_quiver_items):
                try: self.view.removeItem(itm)
                except Exception: pass
            self.quiver_items = []
            self.wind_quiver_items = []
            # keep batched ArrowField items persistent and let _draw_quivers setData() update them
            # this avoids remove/add flicker during refresh
            # draw now (but don't reload environmental datasets if already loaded)
            if not self.sim._env_loaded:
                self._load_environmental_and_quivers()
            else:
                # bypass debounce when user requests a refresh
                self._draw_quivers(force=True)
        except Exception as e:
            print(f"[ShipViewer] ✗ Failed to refresh quivers: {e}")

    def _draw_quivers(self, force: bool = False):
        # Draw quivers using currently-loaded environmental functions.
        # debug: log when quiver drawing occurs
        try:
            ts = datetime.utcnow().isoformat()
            print(f"[ShipViewer] _draw_quivers called at {ts}")
        except Exception:
            pass
        # debounce: avoid redrawing too frequently (protect against per-tick calls)
        now = datetime.utcnow().timestamp()
        last = getattr(self, '_last_quiver_draw_ts', 0.0)
        min_interval = getattr(self, 'quiver_min_interval', 0.8)  # seconds
        if (not force) and (now - last < min_interval):
            # skip drawing to avoid UI strobing
            # print debug once
            try:
                if not getattr(self, '_quiver_debounce_warned', False):
                    print(f"[ShipViewer] Skipping quiver draw (debounce {now-last:.3f}s)")
                    self._quiver_debounce_warned = True
            except Exception:
                pass
            return
        self._last_quiver_draw_ts = now
        self._quiver_debounce_warned = False
        now = datetime.utcnow()
        U_full, V_full = self.sim.currents_grid(now)
        W_full, Vw_full = self.sim.wind_grid(now)
        ll2utm = Transformer.from_crs("EPSG:4326", self.sim.crs_utm, always_xy=True)
        X, Y = ll2utm.transform(self.sim._quiver_lon, self.sim._quiver_lat)

        # Diagnostic: print shapes to help debug mismatches between OFS outputs and our grid
        try:
            print(f"[ShipViewer][debug] X.shape={getattr(X,'shape',None)} Y.shape={getattr(Y,'shape',None)}")
            print(f"[ShipViewer][debug] U_full.shape={getattr(U_full,'shape',None)} V_full.shape={getattr(V_full,'shape',None)}")
            print(f"[ShipViewer][debug] W_full.shape={getattr(W_full,'shape',None)} Vw_full.shape={getattr(Vw_full,'shape',None)}")
        except Exception:
            pass

        # Additional diagnostics: warn if arrays are empty or filled with zeros
        try:
            def _is_all_zero(a):
                arr = np.asarray(a)
                if arr.size == 0:
                    return True
                try:
                    return np.allclose(arr, 0.0)
                except Exception:
                    return False
            if _is_all_zero(U_full) and _is_all_zero(V_full):
                print(f"[ShipViewer][warn] currents grid is all zeros or empty (sim._env_loaded={getattr(self.sim, '_env_loaded', False)})")
            if _is_all_zero(W_full) and _is_all_zero(Vw_full):
                print(f"[ShipViewer][warn] wind grid is all zeros or empty (sim._env_loaded={getattr(self.sim, '_env_loaded', False)})")
        except Exception:
            pass

        qstep = getattr(self, "quiver_step", 2)
        # optionally auto-clamp based on view
        if getattr(self, 'chk_auto_clamp', None) and self.chk_auto_clamp.isChecked():
            try:
                (vx0, vx1), (vy0, vy1) = self.view.viewRange()
                view_span = max(abs(vx1 - vx0), abs(vy1 - vy0))
                # reduce aggressiveness so arrows remain denser when zoomed out
                auto_step = int(min(max(view_span / (self.sim.zoom * 0.35), 1), 16))
                # Only reduce the step (increase density) when zoomed in; do not
                # increase the step (decrease density) when zoomed out. This
                # prevents arrows from becoming sparser as you zoom out.
                if auto_step < qstep:
                    qstep = auto_step
            except Exception:
                pass

        skip = (slice(None, None, qstep), slice(None, None, qstep))
        # Try to extract subsampled grids; be defensive if shapes mismatch
        try:
            qx, qy = X[skip], Y[skip]
            qu, qv = U_full[skip], V_full[skip]
            Wu, Wv = W_full[skip], Vw_full[skip]
        except Exception:
            # Attempt to coerce flat arrays into the expected (ny,nx) grid
            try:
                ny, nx = X.shape
                qu = np.asarray(U_full).reshape(ny, nx)[skip]
                qv = np.asarray(V_full).reshape(ny, nx)[skip]
                Wu = np.asarray(W_full).reshape(ny, nx)[skip]
                Wv = np.asarray(Vw_full).reshape(ny, nx)[skip]
                qx, qy = X[skip], Y[skip]
            except Exception:
                # give up and use empty arrays to avoid index errors
                qx = np.empty((0,))
                qy = np.empty((0,))
                qu = np.empty((0,))
                qv = np.empty((0,))
                Wu = np.empty((0,))
                Wv = np.empty((0,))

        # -------------------- Land mask & optional smoothing -----------------
        try:
            # Mask points that fall on land using ENC LNDARE polygons (if available)
            if getattr(self, 'chk_mask_land', None) and self.chk_mask_land.isChecked():
                land_gdf = getattr(self.sim, 'enc_data', {}).get('LNDARE')
                if land_gdf is not None and not land_gdf.empty:
                    # Cache prepared union geometry for performance
                    if getattr(self, '_land_prepared', None) is None:
                        try:
                            land_union = unary_union(list(land_gdf.geometry))
                            self._land_prepared = prep(land_union)
                        except Exception:
                            self._land_prepared = None
                    if getattr(self, '_land_prepared', None) is not None and qx.size:
                        pts = [Point(x, y) for x, y in zip(qx.ravel(), qy.ravel())]
                        mask = np.array([not self._land_prepared.contains(p) for p in pts], dtype=bool)
                        mask = mask.reshape(qx.shape)
                        qu = np.where(mask, qu, 0.0)
                        qv = np.where(mask, qv, 0.0)
                        Wu = np.where(mask, Wu, 0.0)
                        Wv = np.where(mask, Wv, 0.0)
        except Exception:
            pass

        # Optional smoothing (display only): simple 3x3 box filter
        try:
            if getattr(self, 'chk_smooth_quivers', None) and self.chk_smooth_quivers.isChecked() and qx.size:
                from scipy.ndimage import uniform_filter
                # apply to U/V and wind U/V
                qu = uniform_filter(qu.astype(float), size=3, mode='nearest')
                qv = uniform_filter(qv.astype(float), size=3, mode='nearest')
                Wu = uniform_filter(Wu.astype(float), size=3, mode='nearest')
                Wv = uniform_filter(Wv.astype(float), size=3, mode='nearest')
        except Exception:
            pass

        # compute approximate grid spacing safely
        try:
            dx = X[0,1] - X[0,0]
        except Exception:
            dx = 1.0
        try:
            dy = Y[1,0] - Y[0,0] if Y.shape[0] > 1 else 0.0
        except Exception:
            dy = 0.0
        ds = np.hypot(dx, dy)
        scale = getattr(self, "quiver_scale", 0.6)
        # Use a single small head fraction so all tips match and are slim
        head_frac = 0.10
        # requested offsets: no offset (collapse winds/currents at same grid point)
        offset_frac_curr = 0.0
        offset_frac_wind = 0.0
        # compute view span (kept for offsets) and for compatibility with older code
        try:
            (vx0, vx1), (vy0, vy1) = self.view.viewRange()
            view_span = max(abs(vx1 - vx0), abs(vy1 - vy0))
        except Exception:
            view_span = getattr(self.sim, 'zoom', 5000)
        default_span = max(getattr(self.sim, 'zoom', 5000), 1.0)
        # larger view_span -> smaller view_scale; clamp to sensible range
        view_scale = default_span / max(view_span, 1e-6)
        view_scale = max(0.25, min(view_scale, 6.0))
        # Prefer screen-pixel based sizing so arrows look consistent across zooms.
        try:
            vp = self.view.viewPixelSize()
            # viewPixelSize returns (x_size_in_data_units_per_pixel, y_size...)
            meters_per_px = max(abs(vp[0]), abs(vp[1]))
        except Exception:
            meters_per_px = max(1e-6, ds * 0.001)

        # Desired arrow size in screen pixels, scaled by user quiver_scale
        desired_px = max(6.0, 18.0 * float(getattr(self, 'quiver_scale', 0.6)))
        shaft = desired_px * meters_per_px
        head_curr = max(desired_px * 0.5 * meters_per_px, 2.0)
        head_wind = max(desired_px * 0.5 * meters_per_px, 2.0)

        # Offsets are specified as fractions of grid spacing (±0.5 grid cell)
        offset_dist_curr = offset_frac_curr * ds
        offset_dist_wind = offset_frac_wind * ds

        # Safety: if head size is a substantial fraction of grid spacing, shrink heads
        try:
            head_max_fraction = 0.35  # at most 35% of grid spacing
            if head_curr > head_max_fraction * ds:
                head_curr = head_max_fraction * ds
            if head_wind > head_max_fraction * ds:
                head_wind = head_max_fraction * ds
        except Exception:
            pass

        # Build batched geometry for currents and winds
        curr_lines = []
        curr_heads = []
        wind_lines = []
        wind_heads = []

        for x0, y0, u, v in zip(qx.ravel(), qy.ravel(), qu.ravel(), qv.ravel()):
            mag = np.hypot(u, v)
            if mag < 1e-3 or np.isnan(mag):
                continue
            ux, uy = u/mag, v/mag
            perp_x, perp_y = -uy, ux
            x0c = x0 - perp_x * offset_dist_curr
            y0c = y0 - perp_y * offset_dist_curr
            x1c = x0c + ux*shaft
            y1c = y0c + uy*shaft
            curr_lines.append((x0c, y0c, x1c, y1c))
            # small triangular head centered at (x1c,y1c)
            base_ang = math.atan2(uy, ux)
            left_ang = base_ang + math.pi - math.pi/8
            right_ang = base_ang + math.pi + math.pi/8
            hx1 = x1c + head_curr * math.cos(left_ang)
            hy1 = y1c + head_curr * math.sin(left_ang)
            hx2 = x1c + head_curr * math.cos(right_ang)
            hy2 = y1c + head_curr * math.sin(right_ang)
            curr_heads.append([QPointF(x1c, y1c), QPointF(hx1, hy1), QPointF(hx2, hy2)])

        for x0, y0, u, v in zip(qx.ravel(), qy.ravel(), Wu.ravel(), Wv.ravel()):
            mag = np.hypot(u, v)
            if mag < 1e-3 or np.isnan(mag):
                continue
            ux, uy = u/mag, v/mag
            perp_x, perp_y = -uy, ux
            x0w = x0 + perp_x * offset_dist_wind
            y0w = y0 + perp_y * offset_dist_wind
            x1w = x0w + ux*shaft
            y1w = y0w + uy*shaft
            wind_lines.append((x0w, y0w, x1w, y1w))
            base_ang = math.atan2(uy, ux)
            left_ang = base_ang + math.pi - math.pi/12
            right_ang = base_ang + math.pi + math.pi/12
            hx1 = x1w + head_wind * math.cos(left_ang)
            hy1 = y1w + head_wind * math.sin(left_ang)
            hx2 = x1w + head_wind * math.cos(right_ang)
            hy2 = y1w + head_wind * math.sin(right_ang)
            wind_heads.append([QPointF(x1w, y1w), QPointF(hx1, hy1), QPointF(hx2, hy2)])

        # Create or update batched ArrowField items
        try:
            if getattr(self, 'arrowfield_currents', None) is None:
                self.arrowfield_currents = ArrowField()
                self.view.addItem(self.arrowfield_currents)
            pen_c = QPen(QColor(0,100,255))
            pen_c.setWidthF(getattr(self, 'quiver_pen_width', 2))
            brush_c = QBrush(QColor(0,100,255))
            self.arrowfield_currents.setData(curr_lines, curr_heads, pen_c, brush_c, z=150)

            if getattr(self, 'arrowfield_winds', None) is None:
                self.arrowfield_winds = ArrowField()
                self.view.addItem(self.arrowfield_winds)
            pen_w = QPen(QColor(160,32,240))
            pen_w.setWidthF(getattr(self, 'quiver_pen_width', 2))
            brush_w = QBrush(QColor(160,32,240))
            self.arrowfield_winds.setData(wind_lines, wind_heads, pen_w, brush_w, z=151)
        except Exception as e:
            # fallback to previous per-item drawing if ArrowField fails
            print(f"[ShipViewer] ArrowField batch draw failed, falling back: {e}")
            # fall back: draw individually (this code path rarely executed)
            for a in self.arrowfield_currents, getattr(self, 'arrowfield_winds', None):
                try:
                    if a is not None:
                        self.view.removeItem(a)
                except Exception:
                    pass
            # simple legacy fallback loop
            for x0, y0, u, v in zip(qx.ravel(), qy.ravel(), qu.ravel(), qv.ravel()):
                mag = np.hypot(u, v)
                if mag < 1e-3 or np.isnan(mag):
                    continue
                ux, uy = u/mag, v/mag
                perp_x, perp_y = -uy, ux
                x0c = x0 - perp_x * offset_dist_curr
                y0c = y0 - perp_y * offset_dist_curr
                x1c = x0c + ux*shaft
                y1c = y0c + uy*shaft
                shaft_line = pg.PlotDataItem([x0c, x1c], [y0c, y1c], pen=pg.mkPen(0, 100, 255, width=getattr(self, 'quiver_pen_width', 3)))
                self.view.addItem(shaft_line)
                self.quiver_items.append(shaft_line)
            for x0, y0, u, v in zip(qx.ravel(), qy.ravel(), Wu.ravel(), Wv.ravel()):
                mag = np.hypot(u, v)
                if mag < 1e-3 or np.isnan(mag):
                    continue
                ux, uy = u/mag, v/mag
                perp_x, perp_y = -uy, ux
                x0w = x0 + perp_x * offset_dist_wind
                y0w = y0 + perp_y * offset_dist_wind
                x1w = x0w + ux*shaft
                y1w = y0w + uy*shaft
                wind_shaft = pg.PlotDataItem([x0w, x1w], [y0w, y1w], pen=pg.mkPen(160,32,240, width=max(1, getattr(self, 'quiver_pen_width', 3)-1)))
                self.view.addItem(wind_shaft)
                self.wind_quiver_items.append(wind_shaft)

    print("[ShipViewer] [OK] Quivers drawn (batched)")

    def _start_simulation(self):
        """
        Called when the user clicks 'Start Simulation'.
        Spawns agents in the core, draws ship polygons, and kicks off the timer.
        """
        print("[PLAY] _start_simulation called")
        
        # Ensure animation timer exists (guard against initialization races)
        try:
            if getattr(self, 'timer', None) is None:
                self.timer = QtCore.QTimer(self)
                try:
                    self.timer.timeout.connect(self._update_frame)
                except Exception:
                    pass
        except Exception:
            pass

        # Load environmental forcing if not already loaded
        if not self.sim._env_loaded:
            print("[ShipViewer] Loading environmental forcing before simulation start...")
            self._load_environmental_and_quivers()
        
        # Hide AIS layer and remove plotted route
        if self.ais_item is not None:
            try:
                self.view.removeItem(self.ais_item)
            except Exception:
                pass
            self.ais_item = None
        # Dim any previously drawn route graphics (support list or single item)
        # instead of removing them so the loaded/persisted route remains visible
        # during simulation as a faint grey guide.
        try:
            route_obj = getattr(self, 'route_item', None)
            dim_pen = pg.mkPen((180,180,180,120), width=1)
            if route_obj is not None:
                if isinstance(route_obj, list):
                    for r in route_obj:
                        try:
                            r.setPen(dim_pen)
                            r.setZValue(5)
                        except Exception:
                            pass
                else:
                    try:
                        route_obj.setPen(dim_pen)
                        route_obj.setZValue(5)
                    except Exception:
                        pass
        except Exception:
            pass
            
        # 1) spawn in the core
        # If no waypoints are defined, do not attempt to spawn (especially for auto-start cases).
        # Allow cases where fewer than n waypoint lists are provided; simulation.spawn
        # will duplicate the last route to fill missing agents.
        if not hasattr(self.sim, 'waypoints') or len(getattr(self.sim, 'waypoints', [])) == 0:
            # Inform the user and enable Start so they can define or load a route.
            try:
                QtWidgets.QMessageBox.information(self, "No route defined", "No route is defined for the simulation. Click 'Define Route' to create one or 'Load Persisted' to apply a saved route.")
            except Exception:
                print("[ShipViewer] No route defined — cannot spawn. Please define or load a route.")
            try:
                if hasattr(self, 'btn_start_sim'):
                    self.btn_start_sim.setEnabled(True)
            except Exception:
                pass
            return
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
        # clear any existing trajectory items (fresh spawn)
        for itm in getattr(self, 'traj_items', []):
            try: self.view.removeItem(itm)
            except Exception: pass
        self.traj_items = []
        for i in range(self.sim.n):
            body = self.sim._ship_base
            poly = ship_polygon(body, color=(200,50,50,180))
            poly.setPos(pos0[0, i], pos0[1, i])
            poly.setRotation(np.degrees(psi0[i]))
            poly.setZValue(30)
            self.view.addItem(poly)
            self.ship_items.append(poly)
            # create a placeholder for a persistent flag marker (hidden until set)
            try:
                flag = QGraphicsEllipseItem(-4, -4, 8, 8)
                flag.setBrush(QBrush(QColor(220, 20, 20, 220)))
                flag.setPen(QPen(QColor(0,0,0,200), 1))
                flag.setZValue(120)
                flag.setVisible(False)
                self.view.addItem(flag)
                self.flag_items.append(flag)
            except Exception:
                self.flag_items.append(None)
            # create an empty trajectory line for this ship
            try:
                tline = pg.PlotCurveItem(x=[], y=[], pen=pg.mkPen('lime', width=1))
                tline.setZValue(20)
                self.view.addItem(tline)
                self.traj_items.append(tline)
            except Exception:
                self.traj_items.append(None)
        # create/clear danger cone items (QGraphicsPolygonItem, orange fill)
        for itm in getattr(self, 'danger_cone_items', []):
            try: self.view.removeItem(itm)
            except Exception: pass
        self.danger_cone_items = []
        for k in range(self.sim.n):
            # Compute danger cone geometry for ship k
            pos = self.sim.pos[:, k]
            psi = self.sim.psi[k]
            length = self.sim.ship.length if hasattr(self.sim.ship, 'length') else 400.0
            width = self.sim.ship.beam if hasattr(self.sim.ship, 'beam') else 60.0
            cone_angle = np.radians(25.0)
            cone_length = length * 2.5
            # Bow point
            bow = pos + length * np.array([np.cos(psi), np.sin(psi)])
            # Left/right cone points
            left = bow + cone_length * np.array([np.cos(psi + cone_angle), np.sin(psi + cone_angle)])
            right = bow + cone_length * np.array([np.cos(psi - cone_angle), np.sin(psi - cone_angle)])
            # Stern
            stern = pos - 0.5 * length * np.array([np.cos(psi), np.sin(psi)])
            cone_poly = QPolygonF([
                QPointF(pos[0], pos[1]),
                QPointF(left[0], left[1]),
                QPointF(right[0], right[1]),
                QPointF(stern[0], stern[1])
            ])
            cone_item = QGraphicsPolygonItem(cone_poly)
            cone_item.setBrush(QBrush(QColor(255,165,0,80)))
            cone_item.setPen(QPen(QColor(255,140,0, 120), 2))
            cone_item.setZValue(10)
            self.view.addItem(cone_item)
            self.danger_cone_items.append(cone_item)
            
        x0, y0 = pos0[0,0], pos0[1,0]
        width = height = self.sim.zoom  # e.g. 5000m
        rect = QtCore.QRectF(x0 - width/2, y0 - height/2, width, height)
        self.view.setRange(rect, padding=0.02)
        print(f"   view centered at {x0:.0f},{y0:.0f} ±{width/2}")
                    
        # 4) start the animation timer (ensure it exists)
        try:
            if getattr(self, 'timer', None) is None:
                self.timer = QtCore.QTimer(self)
                try:
                    self.timer.timeout.connect(self._update_frame)
                except Exception:
                    pass
            self.timer.start(int(self.sim.dt * 1000))
        except Exception as e:
            print(f"[ShipViewer] Failed to start timer: {e}")
        try:
            if hasattr(self, 'btn_start_sim'):
                self.btn_start_sim.setEnabled(False)
        except Exception:
            pass
        # After simulation starts, Pause should be enabled and Play disabled
        try:
            if hasattr(self, 'btn_pause'):
                self.btn_pause.setEnabled(True)
            if hasattr(self, 'btn_play'):
                self.btn_play.setEnabled(False)
        except Exception:
            pass
        self.btn_define_route.setEnabled(False)
        # enable kill switch during simulation
        self.btn_kill_power.setEnabled(True)
        # bring viewer to front when simulation starts
        try:
            self._bring_to_front()
        except Exception:
            pass
        
    def _on_kill_power(self):
        """
        Randomly cut power to one vessel via the simulation's kill switch.
        """
        # Delegate to core simulation method
        self.sim._cut_random_power()

    def _on_pause(self):
        """Pause the simulation timer (freeze simulation state)."""
        try:
            if getattr(self, 'timer', None) is not None and self.timer.isActive():
                self.timer.stop()
                # toggle buttons
                try: self.btn_pause.setEnabled(False)
                except Exception: pass
                try: self.btn_play.setEnabled(True)
                except Exception: pass
        except Exception as e:
            print(f"[ShipViewer] Pause failed: {e}")

    def _on_play(self):
        """Resume the simulation timer if paused."""
        try:
            if getattr(self, 'timer', None) is not None and not self.timer.isActive():
                self.timer.start(int(self.sim.dt * 1000))
                try: self.btn_pause.setEnabled(True)
                except Exception: pass
                try: self.btn_play.setEnabled(False)
                except Exception: pass
        except Exception as e:
            print(f"[ShipViewer] Play failed: {e}")

    def _clear_events(self):
        """Clear recorded collision and allision events from the simulation and GUI."""
        try:
            if hasattr(self, 'sim'):
                self.sim.collision_events = []
                self.sim.allision_events = []
        except Exception:
            pass
        try:
            if getattr(self, 'events_view', None) is not None:
                self.events_view.clear()
        except Exception:
            pass

    def showEvent(self, ev):
        # ensure window is on top when first shown
        try:
            self._bring_to_front()
        except Exception:
            pass
        return super().showEvent(ev)

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
                    item.setZValue(150)
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
                        item.setZValue(151)
                        self.view.addItem(item)

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
                # disconnect from the LayoutWidget scene when that agent is done
                self.plot_widget.scene().sigMouseClicked.disconnect(self._on_map_click)
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
                route_items = []
                for agent_pts in self._route_pts_by_agent:
                    xs, ys = zip(*agent_pts)
                    curve = pg.PlotCurveItem(
                        x=np.array(xs), y=np.array(ys),
                        pen=pg.mkPen('lime', width=2)
                    )
                    curve.setZValue(15)
                    self.view.addItem(curve)
                    route_items.append(curve)
                # persist route graphics so they can be removed later
                try:
                    self.route_item = route_items
                except Exception:
                    pass
                # Persist routes to disk so they survive restarts
                try:
                    self._save_persisted_routes(self.sim.waypoints)
                except Exception as e:
                    print(f"[ShipViewer] Failed to save routes: {e}")
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
        # -> hook into the LayoutWidget scene so clicks on the map get through
        self.plot_widget.scene().sigMouseClicked.connect(self._on_map_click)

    def _save_persisted_routes(self, waypoints_list):
        """Save waypoints (list of agents, each list of [x,y]) to JSON keyed by port."""
        try:
            data = {}
            if os.path.exists(self._routes_path):
                with open(self._routes_path, 'r') as fh:
                    try:
                        data = json.load(fh)
                    except Exception:
                        data = {}
            # Convert numpy arrays to lists
            serial = []
            for agent_pts in waypoints_list:
                serial.append([[float(x), float(y)] for x, y in agent_pts])
            data[self.sim.port_name] = {
                'date': datetime.utcnow().isoformat(),
                'waypoints': serial
            }
            with open(self._routes_path, 'w') as fh:
                json.dump(data, fh)
            print(f"[ShipViewer] Routes saved to {self._routes_path}")
        except Exception as e:
            print(f"[ShipViewer] Failed to persist routes: {e}")

    def _load_persisted_routes(self):
        """Detect persisted routes for this port and stash them for explicit user loading.

        Note: We intentionally do NOT auto-apply persisted routes at startup to avoid
        surprising simulation state changes. The user must click 'Load Persisted' to
        apply the saved waypoints to the current simulation.
        """
        try:
            self._persisted_route_entry = None
            if not os.path.exists(self._routes_path):
                return
            with open(self._routes_path, 'r') as fh:
                data = json.load(fh)
            entry = data.get(self.sim.port_name)
            if not entry:
                return
            wps = entry.get('waypoints', [])
            if not wps:
                return
            # stash the persisted waypoint list (raw lists) for later explicit apply
            self._persisted_route_entry = {
                'date': entry.get('date'),
                'waypoints': wps,
            }
            saved_agents = len(wps)
            print(f"[ShipViewer] Detected persisted route for {self.sim.port_name} (agents={saved_agents}) — click 'Load Persisted' to apply")
        except Exception as e:
            print(f"[ShipViewer] Failed to read persisted routes: {e}")

    def _apply_persisted_route(self):
        """Read persisted routes from disk and apply them to the simulation.

        This is intentionally done on demand to avoid changing simulation state
        at startup. The file is read when the user clicks 'Load Persisted'.
        """
        try:
            if not os.path.exists(self._routes_path):
                QtWidgets.QMessageBox.information(self, "No persisted route", "No persisted route file found.")
                return
            with open(self._routes_path, 'r') as fh:
                data = json.load(fh)
            entry = data.get(self.sim.port_name)
            if not entry:
                QtWidgets.QMessageBox.information(self, "No persisted route", "No persisted route found for this port.")
                return
            wps = entry.get('waypoints', [])
            if not wps:
                QtWidgets.QMessageBox.information(self, "No persisted route", "No persisted waypoints found.")
                return
            saved_agents = len(wps)
            sim_n = getattr(self.sim, 'n', 0)
            if saved_agents != sim_n:
                QtWidgets.QMessageBox.warning(self, "Agent count mismatch", f"Saved route has {saved_agents} agents but the simulation has {sim_n}. Adjust agent count or create a new route.")
                print(f"[ShipViewer] Persisted route agent-count mismatch (saved={saved_agents}, sim.n={sim_n}); refusing to apply")
                return
            # draw and set sim.waypoints
            route_items = []
            for agent_pts in wps:
                xs, ys = zip(*agent_pts)
                curve = pg.PlotCurveItem(x=np.array(xs), y=np.array(ys), pen=pg.mkPen((180,180,180,150), width=1))
                curve.setZValue(15)
                self.view.addItem(curve)
                route_items.append(curve)
            self.route_item = route_items
            # Detect coordinate units: if waypoints look like lon/lat (deg), convert to UTM
            try:
                first_pt = wps[0][0]
                x0, y0 = float(first_pt[0]), float(first_pt[1])
                need_transform = (abs(x0) <= 180.0 and abs(y0) <= 90.0)
            except Exception:
                need_transform = False

            if need_transform:
                try:
                    from pyproj import Transformer
                    ll2utm = Transformer.from_crs("EPSG:4326", self.sim.crs_utm, always_xy=True)
                    converted = []
                    failed = False
                    for agent_pts in wps:
                        conv = []
                        for x, y in agent_pts:
                            # try normal (lon, lat)
                            try:
                                ux, uy = ll2utm.transform(float(x), float(y))
                            except Exception:
                                # try swapped (lat, lon)
                                try:
                                    ux, uy = ll2utm.transform(float(y), float(x))
                                    if self.sim.verbose:
                                        print("[ShipViewer] Converted point using swapped (lat,lon) ordering")
                                except Exception:
                                    ux, uy = np.nan, np.nan
                            conv.append(np.array([ux, uy]))
                        converted.append(conv)
                    # sanity-check converted coordinates
                    bad = False
                    for agent in converted:
                        for pt in agent:
                            try:
                                xx, yy = float(pt[0]), float(pt[1])
                                if not (np.isfinite(xx) and np.isfinite(yy)):
                                    bad = True
                                    break
                            except Exception:
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        QtWidgets.QMessageBox.warning(self, "Invalid route", "Persisted route conversion failed; contains invalid coordinate values; cannot apply.")
                        print(f"[ShipViewer] Persisted route conversion produced non-finite coords; refusing to apply")
                        return
                    self.sim.waypoints = converted
                    print(f"[ShipViewer] Converted persisted waypoints from lon/lat → {self.sim.crs_utm} before applying")
                except Exception as e:
                    print(f"[ShipViewer] Failed to convert persisted waypoints to UTM: {e}; applying raw values")
                    self.sim.waypoints = [[np.array(p) for p in agent_pts] for agent_pts in wps]
            else:
                # Validate raw numeric values before assigning
                bad = False
                for agent_pts in wps:
                    for p in agent_pts:
                        try:
                            x, y = float(p[0]), float(p[1])
                            if not (np.isfinite(x) and np.isfinite(y)):
                                bad = True
                                break
                        except Exception:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    QtWidgets.QMessageBox.warning(self, "Invalid route", "Persisted route contains invalid coordinate values; cannot apply.")
                    print(f"[ShipViewer] Persisted route contains non-finite coordinates; refusing to apply")
                    return
                self.sim.waypoints = [[np.array(p) for p in agent_pts] for agent_pts in wps]
            try:
                if hasattr(self, 'btn_start_sim'):
                    self.btn_start_sim.setEnabled(True)
            except Exception:
                pass
            try:
                self._status_label.setText("Loaded persisted route — ready to simulate.")
            except Exception:
                pass
            print(f"[ShipViewer] Applied persisted routes for {self.sim.port_name}")
        except Exception as e:
            print(f"[ShipViewer] Failed to apply persisted routes: {e}")

    # ----- Named route manager handlers ---------------------------------
    def _save_route_as_dialog(self):
        """Prompt for a name and save the current waypoints as a named route."""
        try:
            # ensure there is a route to save
            wps = getattr(self.sim, 'waypoints', None)
            if not wps:
                QtWidgets.QMessageBox.information(self, "No route", "No route is defined to save.")
                return
            name, ok = QtWidgets.QInputDialog.getText(self, "Save Route As", "Route name:")
            if not ok or not name:
                return
            # load existing named routes
            data = {}
            try:
                if os.path.exists(self._named_routes_path):
                    with open(self._named_routes_path, 'r') as fh:
                        data = json.load(fh)
            except Exception:
                data = {}
            port_bucket = data.get(self.sim.port_name, {})
            # serialize waypoints to pure python lists
            serial = []
            for agent_pts in wps:
                serial.append([[float(x), float(y)] for x, y in agent_pts])
            port_bucket[name] = {
                'date': datetime.utcnow().isoformat(),
                'waypoints': serial
            }
            data[self.sim.port_name] = port_bucket
            with open(self._named_routes_path, 'w') as fh:
                json.dump(data, fh)
            # update list widget
            try:
                self.lst_routes.addItem(name)
            except Exception:
                pass
            print(f"[ShipViewer] Saved named route '{name}' for {self.sim.port_name}")
        except Exception as e:
            print(f"[ShipViewer] Failed to save named route: {e}")

    def _load_selected_named_route(self):
        """Load the selected named route from disk and draw it into the view."""
        try:
            it = self.lst_routes.currentItem()
            if it is None:
                QtWidgets.QMessageBox.information(self, "No selection", "Please select a saved route to load.")
                return
            name = it.text()
            if not os.path.exists(self._named_routes_path):
                QtWidgets.QMessageBox.warning(self, "Not found", "Named routes file not found.")
                return
            with open(self._named_routes_path, 'r') as fh:
                data = json.load(fh)
            port_bucket = data.get(self.sim.port_name, {})
            entry = port_bucket.get(name)
            if not entry:
                QtWidgets.QMessageBox.warning(self, "Missing", f"Route '{name}' not found for this port.")
                return
            wps = entry.get('waypoints', [])
            if not wps:
                QtWidgets.QMessageBox.information(self, "Empty", "Selected route contains no waypoints.")
                return
            # clear existing route graphics
            try:
                if getattr(self, 'route_item', None) is not None:
                    if isinstance(self.route_item, list):
                        for r in self.route_item:
                            try: self.view.removeItem(r)
                            except Exception: pass
                    else:
                        try: self.view.removeItem(self.route_item)
                        except Exception: pass
            except Exception:
                pass
            # draw loaded route(s)
            route_items = []
            for agent_pts in wps:
                xs, ys = zip(*agent_pts)
                curve = pg.PlotCurveItem(x=np.array(xs), y=np.array(ys), pen=pg.mkPen((180,180,180,150), width=1))
                curve.setZValue(15)
                self.view.addItem(curve)
                route_items.append(curve)
            self.route_item = route_items
            # set sim.waypoints so Start can use them
            try:
                self.sim.waypoints = [[np.array(p) for p in agent_pts] for agent_pts in wps]
            except Exception:
                pass
            self.btn_start_sim.setEnabled(True)
            self._status_label.setText(f"Loaded named route '{name}' — ready to simulate.")
            print(f"[ShipViewer] Loaded named route '{name}'")
        except Exception as e:
            print(f"[ShipViewer] Failed to load named route: {e}")

    def _delete_selected_named_route(self):
        """Delete the selected named route from both disk and the list widget."""
        try:
            it = self.lst_routes.currentItem()
            if it is None:
                QtWidgets.QMessageBox.information(self, "No selection", "Please select a saved route to delete.")
                return
            name = it.text()
            resp = QtWidgets.QMessageBox.question(self, "Delete Route", f"Delete named route '{name}'?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if resp != QtWidgets.QMessageBox.Yes:
                return
            if not os.path.exists(self._named_routes_path):
                QtWidgets.QMessageBox.warning(self, "Not found", "Named routes file not found.")
                return
            with open(self._named_routes_path, 'r') as fh:
                data = json.load(fh)
            port_bucket = data.get(self.sim.port_name, {})
            if name in port_bucket:
                del port_bucket[name]
                data[self.sim.port_name] = port_bucket
                with open(self._named_routes_path, 'w') as fh:
                    json.dump(data, fh)
            # remove from list widget
            try:
                row = self.lst_routes.currentRow()
                self.lst_routes.takeItem(row)
            except Exception:
                pass
            print(f"[ShipViewer] Deleted named route '{name}'")
        except Exception as e:
            print(f"[ShipViewer] Failed to delete named route: {e}")

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
        
        # ── update time, wind, and current labels ───────────────────────────
        # Time
        self.lbl_time.setText(f"Time: {self.sim.t:.2f} s")

        # get primary ship’s lon/lat
        lon0, lat0 = self.sim._utm_to_ll.transform(
            self.sim.pos[0,0], self.sim.pos[1,0]
        )

        # Wind at ship #0 — sampler output shape can vary (N,2) or (2,N) etc.
        try:
            wres = self.sim.wind_fn(np.array([lon0]), np.array([lat0]), datetime.now(timezone.utc))
            warr = np.asarray(wres)
            # handle (N,2) => warr[0] is [u,v]
            if warr.ndim == 2 and warr.shape[1] == 2:
                wu, wv = float(warr[0, 0]), float(warr[0, 1])
            # handle (2,N) => warr[:,0] is [u,v]
            elif warr.ndim == 2 and warr.shape[0] == 2:
                wu, wv = float(warr[0, 0]), float(warr[1, 0])
            else:
                flat = warr.ravel()
                if flat.size >= 2:
                    wu, wv = float(flat[0]), float(flat[1])
                else:
                    wu, wv = 0.0, 0.0
        except Exception:
            wu, wv = 0.0, 0.0
        w_speed = np.hypot(wu, wv)
        w_dir   = int((np.degrees(np.arctan2(wv, wu)) + 360) % 360)
        self.lbl_wind.setText(f"Wind: {w_speed:.2f} m/s, {w_dir}°")

        # Current at ship #0 — normalize sampler output similarly to wind
        try:
            cres = self.sim.current_fn(np.array([lon0]), np.array([lat0]), datetime.now(timezone.utc))
            carr = np.asarray(cres)
            if carr.ndim == 2 and carr.shape[1] == 2:
                cu, cv = float(carr[0, 0]), float(carr[0, 1])
            elif carr.ndim == 2 and carr.shape[0] == 2:
                cu, cv = float(carr[0, 0]), float(carr[1, 0])
            else:
                flat = carr.ravel()
                if flat.size >= 2:
                    cu, cv = float(flat[0]), float(flat[1])
                else:
                    cu, cv = 0.0, 0.0
        except Exception:
            cu, cv = 0.0, 0.0
        c_speed = np.hypot(cu, cv)
        c_dir   = int((np.degrees(np.arctan2(cv, cu)) + 360) % 360)
        self.lbl_current.setText(f"Current: {c_speed:.2f} m/s, {c_dir}°")

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
        # ─────────────────────────────────────────────────────────────────
        # apply a softened attenuation of environment sampling for UI controller
        # (do NOT negate: samplers already return the drift vector in earth-frame
        #  that points in the direction the medium is pushing the ship)
        orig_wind_fn    = self.sim.wind_fn
        orig_current_fn = self.sim.current_fn
        drift_gain = 0.6   # tune between 0 (no comp) and 1 (full comp)
        self.sim.wind_fn    = lambda lon, lat, now: drift_gain * orig_wind_fn(lon, lat, now)
        self.sim.current_fn = lambda lon, lat, now: drift_gain * orig_current_fn(lon, lat, now)

        hd_cmds, sp_cmds, rud_cmds = self.sim._compute_controls_and_update(
            self.sim.state[[0,1,3]], t)

        # restore original environment‐sampling functions
        self.sim.wind_fn    = orig_wind_fn
        self.sim.current_fn = orig_current_fn
        
        
        
        # disable rudder for vessels that have lost power
        # (recognize cut-power by commanded_rpm == 0)
        cut_mask = (self.sim.ship.commanded_rpm == 0)
        rud_cmds = np.where(cut_mask, 0.0, rud_cmds)

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

        t = self.sim.t
        # 1) Ship–ship collisions
        for i in range(self.sim.n):
            poly_i = self.sim._current_hull_poly(i)
            for j in range(i+1, self.sim.n):
                poly_j = self.sim._current_hull_poly(j)
                if poly_i.intersects(poly_j):
                    inter = poly_i.intersection(poly_j)
                    if inter.area > self.sim.collision_tol_area:
                        # cut power on both vessels
                        self.sim.ship.cut_power(i)
                        self.sim.ship.cut_power(j)
                        print(f"Collision: Ships {i}&{j} at t={t:.2f}s")
                        
        # 2) Ship–land allisions
        for i in range(self.sim.n):
            poly = self.sim._current_hull_poly(i)
            for land in self.sim.waterway.geometry:
                if poly.intersects(land):
                    inter = poly.intersection(land)
                    # cut power and log structured allision event (include heading & yaw-rate)
                    try:
                        self.sim.ship.cut_power(i)
                    except Exception:
                        pass
                    try:
                        from emergent.ship_abm.simulation_core import log as sim_log
                        hd_deg = np.degrees(self.sim.psi[i])
                        r_meas = getattr(self.sim, 'r_meas', None)
                        r_val = float(r_meas[i]) if (r_meas is not None) else float('nan')
                        sim_log.warning('[ALLISION] ship=%s t=%5.2f area=%s bounds=%s hd_deg=%5.1f yaw_rate_deg_s=%5.2f',
                                        i, t, getattr(inter, 'area', 0.0), getattr(inter, 'bounds', None), hd_deg, np.degrees(r_val))
                    except Exception:
                        print(f"Allision: Ship{i} with land at t={t:.2f}s hd={np.degrees(self.sim.psi[i]):.1f} r={np.degrees(getattr(self.sim, 'r_meas', [0])[i]):.2f}deg/s")
        
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
            
        # sample environmental at the primary ship for quick terminal diagnostics
        try:
            lon0, lat0 = self.sim._utm_to_ll.transform(self.sim.pos[0,0], self.sim.pos[1,0])
            wu, wv = self.sim.wind_fn(np.array([lon0]), np.array([lat0]), datetime.now(timezone.utc))[0]
            cu, cv = self.sim.current_fn(np.array([lon0]), np.array([lat0]), datetime.now(timezone.utc))[0]
            w_speed = np.hypot(wu, wv); w_dir = int((np.degrees(np.arctan2(wv, wu)) + 360) % 360)
            c_speed = np.hypot(cu, cv); c_dir = int((np.degrees(np.arctan2(cv, cu)) + 360) % 360)
            print(f"▶ tick t={self.sim.t:.2f}, pos={self.sim.pos[:,0] if self.sim.pos is not None else 'N/A'}; "
                  f"wind={w_speed:.2f}m/s@{w_dir}°; current={c_speed:.2f}m/s@{c_dir}°")
        except Exception:
            # fallback to plain tick if sampling fails
            print(f"▶ tick t={self.sim.t:.2f}, pos={self.sim.pos[:,0] if self.sim.pos is not None else 'N/A'}")

        # ── Clear previous annotations
        # NOTE: avoid removing and re-adding PlotItems each tick to prevent scene churn.
        # We'll reuse existing items and update their data/positions.

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
            rud_cmd_deg = np.degrees(rud_i)        # new: δcmd in degrees
            # 1) Rudder line at the stern (length=0.1·L, tilt=ψ+π+δ+signδ·π/2)
            stern_offset = (L/2) + 0.05 * L
            stern_x = x - stern_offset * np.cos(curr_hd)
            stern_y = y - stern_offset * np.sin(curr_hd)
            tilt     = np.sign(rud_i) * (np.pi/2)
            rud_end_x = stern_x - 0.1 * L * np.cos(curr_hd + np.pi + rud_i + tilt)
            rud_end_y = stern_y - 0.1 * L * np.sin(curr_hd + np.pi + rud_i + tilt)
            # reuse or create rudder line
            if i < len(self.rudder_items):
                rud_line = self.rudder_items[i]
                try:
                    rud_line.setData(x=[stern_x, rud_end_x], y=[stern_y, rud_end_y])
                except Exception:
                    # fallback to replacing item
                    try: self.view.removeItem(rud_line)
                    except Exception: pass
                    rud_line = pg.PlotCurveItem(x=[stern_x, rud_end_x], y=[stern_y, rud_end_y], pen=pg.mkPen('yellow', width=2))
                    rud_line.setZValue(100)
                    self.view.addItem(rud_line)
                    self.rudder_items[i] = rud_line
            else:
                rud_line = pg.PlotCurveItem(x=[stern_x, rud_end_x], y=[stern_y, rud_end_y], pen=pg.mkPen('yellow', width=2))
                rud_line.setZValue(100)
                self.view.addItem(rud_line)
                self.rudder_items.append(rud_line)

            # 2) Desired-heading line at the bow (length=0.2·L)
            bow_x = x + (L/2) * np.cos(curr_hd)
            bow_y = y + (L/2) * np.sin(curr_hd)
            dx_hd = 0.2 * L * np.cos(cmd_hd)
            dy_hd = 0.2 * L * np.sin(cmd_hd)
            # reuse or create heading line
            if i < len(self.heading_items):
                hd_line = self.heading_items[i]
                try:
                    hd_line.setData(x=[bow_x, bow_x + dx_hd], y=[bow_y, bow_y + dy_hd])
                except Exception:
                    try: self.view.removeItem(hd_line)
                    except Exception: pass
                    hd_line = pg.PlotCurveItem(x=[bow_x, bow_x + dx_hd], y=[bow_y, bow_y + dy_hd], pen=pg.mkPen('red', width=2))
                    self.view.addItem(hd_line)
                    self.heading_items[i] = hd_line
            else:
                hd_line = pg.PlotCurveItem(x=[bow_x, bow_x + dx_hd], y=[bow_y, bow_y + dy_hd], pen=pg.mkPen('red', width=2))
                self.view.addItem(hd_line)
                self.heading_items.append(hd_line)

            # 2b) Bow -> next-waypoint line (if a goal exists)
            try:
                goal = None
                if hasattr(self.sim, 'goals') and self.sim.goals is not None:
                    garr = np.atleast_2d(self.sim.goals)
                    # assume goals shape (2, n) or (n, 2)
                    if garr.shape[0] == 2 and garr.shape[1] >= i+1:
                        goal = garr[:, i]
                    elif garr.shape[1] == 2 and garr.shape[0] >= i+1:
                        goal = garr[i, :]
                if goal is not None:
                    gx, gy = float(goal[0]), float(goal[1])
                    # reuse or create bow->goal line
                    if i < len(self.bow_goal_items):
                        bg_line = self.bow_goal_items[i]
                        try:
                            bg_line.setData(x=[bow_x, gx], y=[bow_y, gy])
                        except Exception:
                            try: self.view.removeItem(bg_line)
                            except Exception: pass
                            bg_line = pg.PlotCurveItem(x=[bow_x, gx], y=[bow_y, gy], pen=pg.mkPen('cyan', width=1, style=QtCore.Qt.DashLine))
                            bg_line.setZValue(90)
                            self.view.addItem(bg_line)
                            self.bow_goal_items[i] = bg_line
                else:
                    # hide or clear existing item
                    if i < len(self.bow_goal_items):
                        try:
                            self.bow_goal_items[i].setData(x=[], y=[])
                        except Exception:
                            pass
            except Exception:
                pass

            # 2c) Danger cone polygon (triangle) originating at bow
            # Use the *commanded* heading (cmd_hd) for the cone direction so the
            # visual shows the avoidance/desired heading rather than the current
            # instantaneous heading. Color the cone by role to make overlapping
            # situations obvious (give_way=orange, stand_on=green, neutral=gray).
            try:
                half_ang = np.radians(30.0)
                cone_len = max(500.0, 4.0 * L)
                p0x, p0y = bow_x, bow_y
                # prefer commanded heading if available (hd command may be smoother)
                cone_dir = float(cmd_hd) if cmd_hd is not None else float(curr_hd)
                p1x = p0x + cone_len * math.cos(cone_dir + half_ang)
                p1y = p0y + cone_len * math.sin(cone_dir + half_ang)
                p2x = p0x + cone_len * math.cos(cone_dir - half_ang)
                p2y = p0y + cone_len * math.sin(cone_dir - half_ang)

                # update or create the QGraphicsPolygonItem for the cone so it
                # follows the ship each frame
                if i < len(self.danger_cone_items):
                    cone_item = self.danger_cone_items[i]
                    poly = QPolygonF([QPointF(p0x, p0y), QPointF(p1x, p1y), QPointF(p2x, p2y), QPointF(p0x, p0y)])
                    try:
                        cone_item.setPolygon(poly)
                    except Exception:
                        # in case the item was removed or invalid, recreate it
                        try: self.view.removeItem(cone_item)
                        except Exception: pass
                        cone_item = QGraphicsPolygonItem(poly)
                        cone_item.setZValue(10)
                        self.view.addItem(cone_item)
                        self.danger_cone_items[i] = cone_item

                    # color by role
                    role = roles[i] if (i < len(roles)) else 'neutral'
                    if role == 'give_way':
                        brush_col = QColor(255, 140, 0, 120)   # orange
                        pen_col = QColor(220, 110, 0, 200)
                    elif role == 'stand_on':
                        brush_col = QColor(0, 200, 80, 120)    # green
                        pen_col = QColor(0, 160, 64, 200)
                    else:
                        brush_col = QColor(180, 180, 180, 60)  # gray
                        pen_col = QColor(120, 120, 120, 120)
                    try:
                        cone_item.setBrush(QBrush(brush_col))
                        cone_item.setPen(QPen(pen_col, 2))
                    except Exception:
                        pass
            except Exception:
                # don't let the whole frame fail for one ship's cone
                pass

            # 3) Persistent flagged_give_way marker (UI-visible until acknowledged)
            try:
                if i < len(self.flag_items) and getattr(self.sim.ship, 'flagged_give_way', None) is not None:
                    flag_item = self.flag_items[i]
                    if flag_item is not None:
                        # place just aft of the bow (so it doesn't overlap hull)
                        fx = x + (L/2 + 0.25 * L) * math.cos(curr_hd)
                        fy = y + (L/2 + 0.25 * L) * math.sin(curr_hd)
                        flag_item.setPos(fx, fy)
                        flag_item.setVisible(bool(self.sim.ship.flagged_give_way[i]))
            except Exception:
                pass

            # 3) Text label --------------------------------------------------
            r_dps = np.degrees(r_arr[i])        # deg s-¹ is what mariners think in

            # choose font colour & size based on mode
            colour = "black" if getattr(self.sim, "test_mode", None) == "zigzag" else "white"
            html = (
                "<span style='font-family:Courier New; "
                f"font-size:14px; color:{colour};'>"
                f"u {u_arr[i]:5.1f}  v {v_arr[i]:5.1f}  r {r_dps:7.3f}<br>"
                f"thr {thr_pct[i]:3.0f}%  δcmd {rud_cmd_deg:5.1f}°  δ {rud_deg[i]:5.1f}°<br>"
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
            # update trajectory line from sim.history if present
            try:
                if i < len(self.traj_items) and self.traj_items[i] is not None:
                    traj = np.array(self.sim.history[i]) if hasattr(self.sim, 'history') else np.empty((0,2))
                    if traj.size:
                        self.traj_items[i].setData(x=traj[:,0], y=traj[:,1])
                    else:
                        self.traj_items[i].setData(x=[], y=[])
            except Exception:
                pass
            
        # assume you’ve done something like:
        (xmin, xmax), (ymin, ymax) = self.view.viewRange()

        # Update events panel from simulation (non-blocking)
        try:
            if getattr(self, 'events_view', None) is not None:
                evs = getattr(self.sim, 'collision_events', []) or []
                alls = getattr(self.sim, 'allision_events', []) or []
                lines = []
                for e in evs:
                    # show brief summary per collision
                    lines.append(f"Collision t={e.get('t',0):.1f}s i={e.get('i')} j={e.get('j')} rel_speed={e.get('rel_speed_m_s',0):.2f} m/s")
                for e in alls:
                    lines.append(f"Allision t={e.get('t',0):.1f}s i={e.get('i')} area={e.get('contact_area',0):.1f} m^2")
                self.events_view.setPlainText('\n'.join(lines))
        except Exception:
            pass
        # update the ROI to match
        self.extent_roi.setPos((xmin, ymin))
        self.extent_roi.setSize((xmax - xmin, ymax - ymin))

        # ── Quivers: draw only on-demand (user clicks Refresh or when env first loads)
        # The per-tick quiver redrawing was creating heavy UI load and strobing; skip it here.
        # Instead, update compact wind/current vane in the control panel below.

        # Update compact wind/current vane (single arrow) for ship #0
        try:
            lon0, lat0 = self.sim._utm_to_ll.transform(self.sim.pos[0,0], self.sim.pos[1,0])
            wu, wv = self.sim.wind_fn(np.array([lon0]), np.array([lat0]), datetime.utcnow())[0]
            cu, cv = self.sim.current_fn(np.array([lon0]), np.array([lat0]), datetime.utcnow())[0]
            w_speed = np.hypot(wu, wv)
            w_dir = (np.degrees(np.arctan2(wv, wu)) + 360) % 360
            c_speed = np.hypot(cu, cv)
            c_dir = (np.degrees(np.arctan2(cv, cu)) + 360) % 360
            # update labels if present
            try:
                self.lbl_wind.setText(f"Wind: {w_speed:.2f} m/s @ {int(w_dir)}°")
                self.lbl_current.setText(f"Current: {c_speed:.2f} m/s @ {int(c_dir)}°")
            except Exception:
                pass
            # update simple vane graphics if available
            if getattr(self, 'wind_arrow', None) is not None:
                # rotate wind arrow (pyqtgraph TextItem / arrow widget expects degrees CCW)
                try:
                    self.wind_arrow.setRotation(-w_dir)
                except Exception:
                    pass
            if getattr(self, 'current_arrow', None) is not None:
                try:
                    self.current_arrow.setRotation(-c_dir)
                except Exception:
                    pass
        except Exception:
            # sampling failed; ignore (we'll still run simulation)
            pass

        # Quivers are not updated every timestep to avoid UI strobing.
        # Use the 'Refresh Quivers Now' button to redraw the full quiver field on demand.


        # # ── UPDATE ENVIRONMENT PANEL ───────────────────────────────────────
        # self.lbl_time.setText(f"Time: {self.sim.t:.2f} s")
        # # wind stats & vane
        # wv = playful_wind(self.sim.state, self.sim.t)
        # wx, wy = wv[:,0]
        # ws = np.hypot(wx, wy)
        # wd = np.degrees(np.arctan2(wy, wx)) % 360
        # self.lbl_wind.setText(f"Wind: {ws:.2f} m/s, {wd:.0f}°")
        # # ensure vane points even if speed zero
        # self.wind_arrow.setRotation(-wd)
        # # current stats & vane
        # lon_lbl, lat_lbl = self.sim._utm_to_ll.transform(
        #     self.sim.pos[0, 0], self.sim.pos[1, 0]
        # )
        # cv = self.sim.current_fn(
        #     np.array([lon_lbl]),
        #     np.array([lat_lbl]),
        #     datetime.now(timezone.utc)
        # ).T   # (2,1)        
        # cx, cy = cv[:,0]
        # cs = np.hypot(cx, cy)
        # cd = np.degrees(np.arctan2(cy, cx)) % 360
        # print(f"[DEBUG] t={self.sim.t:.1f}s  lon={lon_lbl:.4f} lat={lat_lbl:.4f}  "
        #       f"u={cx:.3f} m/s  v={cy:.3f} m/s")

        # self.lbl_current.setText(f"Current: {cs:.2f} m/s, {cd:.0f}°")
        # self.current_arrow.setRotation(-cd)

    def _run_production(self):
        n_iter = self.spin_iterations.value()
        # if only one, delegate to manual mode (user presses failure button)
        if n_iter == 1:
            self._start_simulation()
            return
        # setup SQLite DB for results
        conn = sqlite3.connect('production_results.db')
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS results
               (iter INTEGER, timestamp TEXT, failure_agent INTEGER,
                failure_time REAL, metrics TEXT)'''
        )
        conn.commit()
        # batch runs
        for it in range(n_iter):
            # instantiate fresh simulation
            self._start_simulation()  # sets up self.sim
            # run until proximity‐triggered failure
            failed = False
            t = 0.0
            dt = self.sim.dt
            while t < self.sim.steps:
                self.sim.step()
                pos = self.sim.positions  # shape (2, n)
                if not failed:
                    # pairwise distances vectorized
                    delta = pos[:, :, None] - pos[:, None, :]
                    d2 = (delta**2).sum(axis=0)
                    np.fill_diagonal(d2, np.inf)
                    if d2.min() < 1e6:  # (1000 m)^2
                        idx = np.random.randint(self.sim.n)
                        self.sim.ship.cut_power(idx)
                        fail_time = t
                        failure_agent = idx
                        failed = True
                t += dt
            # gather metrics
            metrics = {
                'final_pos': self.sim.positions.tolist(),
                'collisions': getattr(self.sim, 'collision_events', [])
            }
            # insert into DB
            c.execute(
                'INSERT INTO results VALUES (?,?,?,?,?)',
                (it, datetime.now().isoformat(),
                 failure_agent if failed else None,
                 fail_time if failed else None,
                 json.dumps(metrics))
            )
            conn.commit()
        conn.close()
        QtWidgets.QMessageBox.information(
            self, 'Production Run', f'Completed {n_iter} iterations')

    def update_inset_rect(self, *args, **kwargs):
        """
        Placeholder for sigRangeChanged handler.
        Called whenever the main view range changes; no-op for now.
        """
        pass

    def _zoom(self, scale_factor: float):
        """Zoom the main view by scale_factor around the current center.

        scale_factor < 1 zooms in; >1 zooms out.
        """
        try:
            (xmin, xmax), (ymin, ymax) = self.view.viewRange()
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            w = (xmax - xmin) * scale_factor
            h = (ymax - ymin) * scale_factor
            rect = QtCore.QRectF(cx - w/2.0, cy - h/2.0, w, h)
            # use setRange to avoid emitting extra scene changes
            self.view.setRange(rect, padding=0.02)
        except Exception:
            pass

    def _center_on_ship(self):
        """Center the view on the primary ship (agent 0) without changing zoom."""
        try:
            if getattr(self, 'ship_items', None) and len(self.ship_items) > 0:
                x = self.sim.pos[0, 0]
                y = self.sim.pos[1, 0]
            else:
                # fallback to port center
                x = 0.5 * (self.sim.minx + self.sim.maxx)
                y = 0.5 * (self.sim.miny + self.sim.maxy)
            (xmin, xmax), (ymin, ymax) = self.view.viewRange()
            w = xmax - xmin
            h = ymax - ymin
            rect = QtCore.QRectF(x - w/2.0, y - h/2.0, w, h)
            self.view.setRange(rect, padding=0.02)
        except Exception:
            pass
