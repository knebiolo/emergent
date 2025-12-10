"""
salmon_viewer_v2.py

A clean, minimal replacement viewer module to be used while we repair the
original `salmon_viewer.py`. This file contains a single-threaded GL mesh
builder (QThread) and a compact `SalmonViewer` QWidget with `launch_viewer`.

Purpose: provide a working baseline so you can run the RL visual training GUI
and verify GL TIN rendering without the mangled original file.
"""
import sys
import logging
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QSlider, QGroupBox, QCheckBox
import os

import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
except Exception:
    gl = None
finally:
    try:
        logger = logging.getLogger(__name__)
        logger.debug('pyqtgraph.opengl available: %s', gl is not None)
    except Exception:
        pass

from contextlib import contextmanager
import traceback


@contextmanager
def maybe_suppress(context: str | None = None):
    try:
        yield
    except Exception as exc:
        ctx = f" ({context})" if context else ""
        try:
            tb = traceback.format_exc()
            logger = logging.getLogger(__name__)
            logger.exception('maybe_suppress caught exception%s: %s', ctx, exc)
            logger.debug('\n%s', tb)
        except Exception:
            try:
                logging.getLogger(__name__).error('maybe_suppress caught exception%s: %s', ctx, exc)
            except Exception:
                pass


if gl is not None:
    try:
        class DebugGLViewWidget(gl.GLViewWidget):
            def paintGL(self, *args, **kwargs):
                try:
                    from PyQt5.QtGui import QOpenGLContext
                    ctx = QOpenGLContext.currentContext()
                    print(f'[DBG_GL] paintGL called; currentContext={ctx}')
                    if ctx is not None:
                        try:
                            fmt = ctx.format()
                            print(f'[DBG_GL] ctx fmt: {fmt.majorVersion()}.{fmt.minorVersion()} profile={fmt.profile()}')
                        except Exception:
                            pass
                except Exception as e:
                    print('[DBG_GL] paintGL context query failed:', e)
                try:
                    super().paintGL(*args, **kwargs)
                except Exception as e:
                    print('[DBG_GL] super().paintGL raised:', e)
    except Exception:
        DebugGLViewWidget = None
else:
    DebugGLViewWidget = None


# PyOpenGL-based FBO renderer removed — prefer the Qt CPU painter renderer below.


# PersistentOffscreenRenderer removed: prefer Qt CPU painter renderer (OffscreenQtFBORenderer).


class OffscreenQtFBORenderer:
    """PyQt-native FBO renderer using QOpenGLFramebufferObject + QPainter.

    This renderer avoids PyOpenGL and uses Qt painting to draw a simple
    triangle mesh projection into an image. It's robust across ANGLE and
    desktop contexts.
    """
    def __init__(self, width=800, height=600):
        # CPU-based painter rendering into a QImage; avoids GL context issues
        self.width = int(width)
        self.height = int(height)
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    def render(self, payload, size=None, bgcolor=(0.94, 0.94, 0.94, 1.0)):
        from PyQt5.QtGui import QPainter, QColor, QImage
        from PyQt5.QtCore import QRectF

        if size is not None:
            w, h = int(size[0]), int(size[1])
            if (w, h) != (self.width, self.height):
                self.width, self.height = w, h

        verts = np.asarray(payload['verts'], dtype=float)
        faces = np.asarray(payload['faces'], dtype=int)
        colors = np.asarray(payload['colors'], dtype=float)

        # paint into a QImage using QPainter (CPU rasterization)
        from PyQt5.QtGui import QImage
        img = QImage(self.width, self.height, QImage.Format_RGBA8888)
        img.fill(0)
        qp = QPainter()
        try:
            if not qp.begin(img):
                raise RuntimeError('QPainter.begin(QImage) failed')
            # fill background
            bg = QColor()
            bg.setRgbF(*bgcolor[:3], bgcolor[3] if len(bgcolor) > 3 else 1.0)
            qp.fillRect(0, 0, self.width, self.height, bg)

            # simple orthographic projection: map XY to image coordinates
            minxy = np.min(verts[:, :2], axis=0)
            maxxy = np.max(verts[:, :2], axis=0)
            center = (minxy + maxxy) / 2.0
            span = max(maxxy - minxy)
            if span <= 0:
                span = 1.0
            sx = (self.width * 0.9) / span
            sy = (self.height * 0.9) / span
            tx = self.width / 2.0 - (center[0] * sx)
            ty = self.height / 2.0 + (center[1] * sy)

            # draw triangles
            from PyQt5.QtGui import QPolygonF
            from PyQt5.QtCore import QPointF
            for tri in faces:
                pts = []
                cols = []
                for idx in tri:
                    x, y = verts[idx, 0], verts[idx, 1]
                    px = x * sx + tx
                    py = -y * sy + ty
                    pts.append((px, py))
                    cols.append(colors[idx])
                # average color for triangle
                meanc = np.clip(np.mean(np.asarray(cols), axis=0), 0.0, 1.0)
                c = QColor()
                c.setRgbF(float(meanc[0]), float(meanc[1]), float(meanc[2]), float(meanc[3] if len(meanc) > 3 else 1.0))
                qp.setBrush(c)
                qp.setPen(c)
                poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
                qp.drawPolygon(poly)
        finally:
            try:
                qp.end()
            except Exception:
                pass
        return img
        rlay = QVBoxLayout(right)
        # Playback controls (match original viewer ordering and widget names)
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self._on_play)
        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_simulation)
        rlay.addWidget(self.play_btn)
        rlay.addWidget(self.pause_btn)
        rlay.addWidget(self.reset_btn)

        # Utility buttons (match original viewer ordering)
        btn_rebuild = QPushButton('Rebuild TIN')
        btn_rebuild.clicked.connect(self.rebuild_tin_action)
        rlay.addWidget(btn_rebuild)
        btn_perim = QPushButton('Toggle Perimeter')
        btn_perim.clicked.connect(self.toggle_perimeter)
        rlay.addWidget(btn_perim)

        # Small visual toggles to match original viewer
        try:
            self.show_dead_cb = QCheckBox('Show Dead')
            self.show_dead_cb.setChecked(False)
            rlay.addWidget(self.show_dead_cb)
            self.show_direction_cb = QCheckBox('Show Direction')
            self.show_direction_cb.setChecked(False)
            rlay.addWidget(self.show_direction_cb)
            self.show_tail_cb = QCheckBox('Show Tail')
            self.show_tail_cb.setChecked(False)
            rlay.addWidget(self.show_tail_cb)
        except Exception:
            pass

        ve_group = QGroupBox('Vertical Exaggeration')
        ve_layout = QVBoxLayout()
        self.ve_label = QLabel('Z Exag: 1.00x')
        ve_layout.addWidget(self.ve_label)
        self.ve_slider = QSlider(Qt.Horizontal)
        self.ve_slider.setMinimum(1)
        self.ve_slider.setMaximum(500)
        self.ve_slider.setValue(100)
        self.ve_slider.valueChanged.connect(lambda v: self.ve_label.setText(f'Z Exag: {v/100:.2f}x'))
        ve_layout.addWidget(self.ve_slider)
        ve_group.setLayout(ve_layout)
        rlay.addWidget(ve_group)

        # assemble main three-column layout: left | center | right
        main.addWidget(left, 1)
        main.addWidget(center, 3)
        main.addWidget(right, 1)

        # Metrics placeholders (ported from original viewer)
        self.mean_speed_label = QLabel('Mean Speed: --')
        self.max_speed_label = QLabel('Max Speed: --')
        self.mean_energy_label = QLabel('Mean Energy: --')
        self.min_energy_label = QLabel('Min Energy: --')
        self.upstream_progress_label = QLabel('Upstream Progress: --')
        self.mean_centerline_label = QLabel('Mean Centerline: --')
        self.mean_passage_delay_label = QLabel('Mean Passage Delay: --')
        self.passage_success_rate_label = QLabel('Passage Success: --')

        # compact metrics layout at the bottom of the right pane
        try:
            metrics_box = QGroupBox('Metrics')
            metrics_layout = QVBoxLayout()
            metrics_layout.addWidget(self.mean_speed_label)
            metrics_layout.addWidget(self.max_speed_label)
            metrics_layout.addWidget(self.mean_energy_label)
            metrics_layout.addWidget(self.min_energy_label)
            metrics_layout.addWidget(self.upstream_progress_label)
            metrics_layout.addWidget(self.mean_centerline_label)
            metrics_layout.addWidget(self.mean_passage_delay_label)
            metrics_layout.addWidget(self.passage_success_rate_label)
            # small extras
            self.mean_nn_dist_label = QLabel('Mean NN Dist: --')
            self.polarization_label = QLabel('Polarization: --')
            metrics_layout.addWidget(self.mean_nn_dist_label)
            metrics_layout.addWidget(self.polarization_label)

            # per-episode metrics tracking helper (checkboxes)
            self._available_episode_metrics = [
                'collision_count', 'mean_upstream_progress', 'mean_upstream_velocity',
                'energy_efficiency', 'mean_passage_delay'
            ]
            self.track_metric_cbs = {}
            def add_label_with_cb(label_widget, metric_key, default_checked=False):
                h = QHBoxLayout()
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(6)
                h.addWidget(label_widget)
                cb = QCheckBox()
                cb.setChecked(default_checked)
                cb.setFixedWidth(22)
                h.addWidget(cb)
                metrics_layout.addLayout(h)
                self.track_metric_cbs[metric_key] = cb

            # Map labels to metric keys (match original viewer)
            # collision_count label (new)
            self.collision_count_label = QLabel('Collision Count: --')
            add_label_with_cb(self.collision_count_label, 'collision_count')
            add_label_with_cb(self.upstream_progress_label, 'mean_upstream_progress')
            self.mean_upstream_velocity_label = QLabel('Mean Upstream Velocity: --')
            add_label_with_cb(self.mean_upstream_velocity_label, 'mean_upstream_velocity')
            add_label_with_cb(self.mean_energy_label, 'energy_efficiency')
            add_label_with_cb(self.mean_passage_delay_label, 'mean_passage_delay')

            # per-episode plot
            self.per_episode_plot = pg.PlotWidget(title='Per-Episode Metrics')
            self.per_episode_plot.setLabel('bottom', 'Episode')
            self.per_episode_plot.setLabel('left', 'Metric Value')
            self.per_episode_plot.setMaximumHeight(220)
            metrics_layout.addWidget(self.per_episode_plot)

            metrics_box.setLayout(metrics_layout)
            # Place metrics box in the left column to match original viewer layout
            try:
                left_layout.addWidget(metrics_box)
            except Exception:
                rlay.addWidget(metrics_box)
        except Exception:
            pass

        # RL status labels
        try:
            self.episode_label = QLabel('Episode: 0 | Timestep: 0')
            self.reward_label = QLabel('Reward: 0.00')
            self.best_reward_label = QLabel('Best: 0.00')
            rlay.addWidget(self.episode_label)
            rlay.addWidget(self.reward_label)
            rlay.addWidget(self.best_reward_label)
        except Exception:
            pass
        # RL training state (port from original viewer)
        try:
            self.current_episode = 0
            self.n_timesteps = getattr(self, 'T', 600)
            self.episode_reward = 0.0
            self.best_reward = float('-inf')
            self.prev_metrics = None
            self.rewards_history = []
            # reward plot placeholder (optional)
            try:
                from pyqtgraph import mkPen
                self.reward_plot = pg.PlotWidget(title='Episode Rewards')
                self.reward_plot.setMaximumHeight(160)
                rlay.addWidget(self.reward_plot)
            except Exception:
                self.reward_plot = None
        except Exception:
            pass

    def _on_play(self):
        self.paused = False
        try:
            self.play_btn.setText('Play')
            self.pause_btn.setText('Pause')
        except Exception:
            pass

    def toggle_perimeter(self):
        try:
            if not hasattr(self, 'perim_visible'):
                self.perim_visible = True
            # toggle
            self.perim_visible = not self.perim_visible
            if getattr(self, 'perim_scatter', None) is None:
                return
            if self.perim_visible:
                try:
                    self.gl_view.addItem(self.perim_scatter)
                except Exception:
                    pass
            else:
                try:
                    self.gl_view.removeItem(self.perim_scatter)
                except Exception:
                    pass
        except Exception:
            pass

    def setup_background(self):
        try:
            print('[RENDER] setup_background start')
            try:
                perim_info = getattr(self.sim, 'perimeter_points', None)
                print(f'[RENDER] sim.perimeter_points type={type(perim_info)}')
                if hasattr(perim_info, 'shape'):
                    print(f'[RENDER] sim.perimeter_points shape={getattr(perim_info, "shape", None)}')
            except Exception:
                pass
            if not (hasattr(self.sim, 'use_hecras') and self.sim.use_hecras and hasattr(self.sim, 'hecras_plan_path')):
                print('[RENDER] No HECRAS configured')
                return
            import h5py
            plan = self.sim.hecras_plan_path
            with h5py.File(plan, 'r') as hdf:
                coords = np.array(hdf['Geometry/2D Flow Areas/2D area/Cells Center Coordinate'][:])
                depth = np.array(hdf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D area/Cell Hydraulic Depth'][0, :])
            # determine sampling parameters
            max_nodes = getattr(self.sim, 'tin_max_nodes', 5000)
            depth_thresh = getattr(self.sim, 'tin_depth_thresh', 0.05)

            mask = depth > depth_thresh
            pts = coords[mask]
            vals = depth[mask]

            # improved spatial sampling (if helper available)
            if sample_evenly is not None and len(pts) > max_nodes:
                pts, vals = sample_evenly(pts, vals, max_nodes=max_nodes, grid_dim=120)
            elif len(pts) > max_nodes:
                idx = np.random.default_rng(0).choice(len(pts), size=max_nodes, replace=False)
                pts = pts[idx]; vals = vals[idx]

            if len(pts) < 3:
                print('[RENDER] Not enough points for TIN')
                return

            # Triangulation will be computed in the background builder thread
            # Use perimeter generated by the simulation (vector-first). Viewer no longer
            # performs its own dry-cell inference — sim is the single source of truth.
            try:
                sim_perim = getattr(self.sim, 'perimeter_points', None)
                # Be tolerant: perimeter_points may be a list, array, or other container.
                if sim_perim is None:
                    self.perimeter_pts = None
                else:
                    try:
                        # If it's a list of pairs or dicts, convert to an Nx2 array
                        if isinstance(sim_perim, list):
                            if len(sim_perim) == 0:
                                self.perimeter_pts = None
                            else:
                                first = sim_perim[0]
                                if isinstance(first, dict):
                                    # common keys might be 'x','y' or 'lon','lat' or '0','1'
                                    xs = []
                                    ys = []
                                    for item in sim_perim:
                                        if 'x' in item and 'y' in item:
                                            xs.append(item['x']); ys.append(item['y'])
                                        elif 'lon' in item and 'lat' in item:
                                            xs.append(item['lon']); ys.append(item['lat'])
                                        elif 0 in item and 1 in item:
                                            xs.append(item[0]); ys.append(item[1])
                                        else:
                                            # try to unpack list-like
                                            try:
                                                a, b = list(item)[:2]
                                                xs.append(a); ys.append(b)
                                            except Exception:
                                                xs.append(np.nan); ys.append(np.nan)
                                    self.perimeter_pts = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
                                else:
                                    # list of numeric pairs
                                    try:
                                        arr = np.asarray(sim_perim, dtype=float)
                                        if arr.ndim == 2 and arr.shape[1] >= 2:
                                            self.perimeter_pts = arr[:, :2]
                                        else:
                                            self.perimeter_pts = None
                                    except Exception:
                                        self.perimeter_pts = None
                        else:
                            # numpy array or other; try to coerce
                            arr = np.asarray(sim_perim)
                            if arr.ndim >= 2 and arr.shape[1] >= 2:
                                self.perimeter_pts = arr[:, :2]
                            else:
                                self.perimeter_pts = None
                    except Exception as e:
                        print(f"[RENDER] Warning: couldn't coerce sim.perimeter_points: {e} (type={type(sim_perim)})")
                        self.perimeter_pts = None
            except Exception:
                self.perimeter_pts = None

            # Even if GL is unavailable, we still run the mesh builder so we
            # can produce a matplotlib-based 2D preview. Only create a GL
            # view if `pyqtgraph.opengl` is available.
            if gl is not None and self.gl_view is None:
                try:
                    if DebugGLViewWidget is not None:
                        self.gl_view = DebugGLViewWidget()
                    else:
                        self.gl_view = gl.GLViewWidget()
                    try:
                        self.gl_view.setMouseTracking(True)
                        vp = getattr(self.gl_view, 'viewport', None)
                        if callable(vp):
                            try:
                                vobj = self.gl_view.viewport()
                                vobj.setMouseTracking(True)
                            except Exception:
                                pass
                        # Encourage a native window backing which many Windows
                        # drivers require for on-screen GL rendering.
                        try:
                            from PyQt5.QtCore import Qt
                            try:
                                self.gl_view.setAttribute(Qt.WA_NativeWindow, True)
                            except Exception:
                                pass
                            try:
                                self.gl_view.setAttribute(Qt.WA_PaintOnScreen, True)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            self.gl_view.setMinimumSize(200, 200)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    parent = self.plot_widget.parent()
                    layout = parent.layout()
                    layout.removeWidget(self.plot_widget)
                    self.plot_widget.hide()
                    layout.addWidget(self.gl_view)
                    # Make GL background light so meshes are visible
                    try:
                        from PyQt5.QtGui import QColor
                        self.gl_view.setBackgroundColor(QColor(240, 240, 240))
                    except Exception:
                        try:
                            self.gl_view.setBackgroundColor((1, 1, 1, 1))
                        except Exception:
                            pass
                    # Add a grid for orientation and visual reference
                    try:
                        grid = gl.GLGridItem()
                        grid.scale(10.0, 10.0, 1.0)
                        try:
                            self._safe_add(grid, mark='grid')
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Debug: print active QOpenGLContext and GL strings
                    try:
                        from PyQt5.QtGui import QOpenGLContext
                        ctx = QOpenGLContext.currentContext()
                        print(f'[GL] QOpenGLContext current: {ctx}')
                        if ctx is not None:
                            fmt = ctx.format()
                            try:
                                print(f"[GL] Context version: {fmt.majorVersion()}.{fmt.minorVersion()} profile={fmt.profile()}")
                            except Exception:
                                pass
                        try:
                            # query GL renderer strings via PyOpenGL if available
                            import OpenGL.GL as GL
                            try:
                                renderer = GL.glGetString(GL.GL_RENDERER)
                                vendor = GL.glGetString(GL.GL_VENDOR)
                                version = GL.glGetString(GL.GL_VERSION)
                                print(f"[GL] Vendor={vendor} Renderer={renderer} Version={version}")
                            except Exception as e:
                                print('[GL] glGetString failed:', e)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # set a safe initial camera position
                    try:
                        self.gl_view.setCameraPosition(distance=200, elevation=20, azimuth=-30)
                    except Exception:
                        try:
                            self.gl_view.setCameraPosition(pos=None, distance=200, elevation=20, azimuth=-30)
                        except Exception:
                            pass
                except Exception:
                    self.gl_view = None

            # pass sim polygon so clipping happens in worker thread
            sim_poly = getattr(self.sim, 'perimeter_polygon', None)
            if getattr(self, 'perimeter_pts', None) is None:
                print('[RENDER] Warning: sim.perimeter_points unavailable or malformed; viewer will still attempt TIN but may have holes')
            builder = _GLMeshBuilder(pts, vals, vert_exag=getattr(self.sim, 'vert_exag', 1.0), poly=sim_poly, parent=self)

            def _on_mesh(payload):
                if 'error' in payload:
                    print('[TIN] builder error', payload['error'])
                    return
                # expose latest payload for external tests/inspection
                try:
                    self.last_mesh_payload = payload
                except Exception:
                    pass
                # payload received and stored at `self.last_mesh_payload`
                verts = payload['verts']; faces = payload['faces']; colors = payload['colors']
                try:
                    print(f'[TIN] payload shapes: verts={getattr(verts,"shape",None)}, faces={getattr(faces,"shape",None)}, colors={getattr(colors,"shape",None)}')
                except Exception:
                    pass
                # If GL is available and we have a GL view, create a GL mesh
                # item; otherwise skip GL rendering and rely on 2D preview.
                if gl is not None and getattr(self, 'gl_view', None) is not None:
                    try:
                        # clipping already handled in worker thread
                        meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
                        mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawEdges=False, shader='shaded')
                        if self.tin_mesh is not None:
                            try:
                                self._safe_remove(self.tin_mesh)
                            except Exception:
                                pass
                        self.tin_mesh = mesh
                        try:
                            # Enable edges so the mesh is visible even if shaded lighting is poor
                            mesh.setGLOptions('opaque')
                            mesh.opts['drawEdges'] = True
                            try:
                                mesh.opts['edgeColor'] = (0.2, 0.2, 0.2, 1.0)
                            except Exception:
                                pass
                            try:
                                self._safe_add(mesh, mark='tin_mesh')
                            except Exception:
                                try:
                                    self.gl_view.addItem(mesh)
                                except Exception as e:
                                    print('[TIN] failed to add mesh to gl_view', e)
                        except Exception as e:
                            print('[TIN] GL mesh creation failed:', e)

                        # Debugging helper: log mesh vs sim agent bounds (non-fatal)
                        try:
                            if hasattr(self, 'inspect_mesh_against_sim'):
                                try:
                                    self.inspect_mesh_against_sim(verts)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Also add a lightweight point overlay (robust fallback)
                        try:
                            if getattr(self, 'tin_points', None) is not None:
                                try:
                                    self._safe_remove(self.tin_points)
                                except Exception:
                                    pass
                            pts3 = verts[:, :3].astype(float)
                            # use a bright, opaque color for points
                            try:
                                pcolors = (1.0, 0.2, 0.2, 1.0)
                            except Exception:
                                pcolors = None
                            self.tin_points = gl.GLScatterPlotItem(pos=pts3, color=pcolors, size=6)
                            try:
                                self._safe_add(self.tin_points, mark='tin_points')
                            except Exception:
                                try:
                                    self.gl_view.addItem(self.tin_points)
                                except Exception as e:
                                    print('[TIN] failed to add scatter overlay:', e)
                        except Exception as e:
                            print('[TIN] failed to add scatter overlay:', e)
                    except Exception as e:
                        print('[TIN] GL mesh overall failed:', e)

                    # Force the GL widget to refresh/raise so it's visible on-screen
                    try:
                        try:
                            self.gl_view.setVisible(True)
                            self.gl_view.show()
                            self.gl_view.raise_()
                            self.gl_view.update()
                            self.gl_view.repaint()
                        except Exception:
                            pass
                        try:
                            geom = self.gl_view.geometry()
                            print(f'[TIN] gl_view visible={self.gl_view.isVisible()}, geometry={geom.getRect() if geom is not None else None}')
                        except Exception:
                            pass
                        try:
                            # list child widgets for debugging
                            parent = self.gl_view.parent()
                            children = parent.findChildren(QtWidgets.QWidget) if parent is not None else []
                            print(f'[TIN] parent {parent}; child_count={len(children)}')
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # fit camera to mesh extents and save a one-time screenshot for verification
                    try:
                        verts_min = np.min(verts[:, :2], axis=0)
                        verts_max = np.max(verts[:, :2], axis=0)
                        center = (verts_min + verts_max) / 2.0
                        size = np.max(verts_max - verts_min)
                        try:
                            self.gl_view.setCameraPosition(pos=QtCore.QVector3D(center[0], center[1], size*1.0), elevation=90, azimuth=0)
                        except Exception:
                            try:
                                self.gl_view.setCameraPosition(distance=max(1.0, size*3.0), elevation=60, azimuth=45)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        out_dir = getattr(self.sim, 'model_dir', None) or '.'
                        out_path = os.path.join(out_dir, 'outputs', 'tin_screenshot.png')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        try:
                            img = self.gl_view.readQImage()
                            img.save(out_path)
                            print(f'[TIN] screenshot saved: {out_path}')
                            try:
                                from datetime import datetime
                                repo_out_dir = os.path.join(os.getcwd(), 'outputs')
                                os.makedirs(repo_out_dir, exist_ok=True)
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                repo_path = os.path.join(repo_out_dir, f'tin_screenshot_{ts}.png')
                                img.save(repo_path)
                                print(f'[TIN] repo screenshot saved: {repo_path}')
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    # No GL available: still save a matplotlib-based preview as a PNG
                    try:
                        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
                        from datetime import datetime
                        # We'll save the matplotlib preview below; create file paths first
                        out_dir = getattr(self.sim, 'model_dir', None) or '.'
                        out_path = os.path.join(out_dir, 'outputs', 'tin_preview.png')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    except Exception:
                        out_path = None

                # Prefer a Qt-native FBO renderer for a preview when GL view is not presenting
                try:
                    preview_img = None
                    # try low-res cached render first to keep background updates fast
                    try:
                        key = (verts.shape[0], faces.shape[0], int(getattr(self, 've_slider', None).value() if getattr(self, 've_slider', None) is not None else 100))
                        if hasattr(self, '_preview_cache') and key in self._preview_cache:
                            preview_img = self._preview_cache[key]
                        else:
                            # generate low-res and cache it
                            if getattr(self, 'qt_fbo_renderer', None) is not None:
                                try:
                                    low = self.qt_fbo_renderer.render({'verts': verts, 'faces': faces, 'colors': colors}, size=(320, 240))
                                    if low is not None:
                                        preview_img = low
                                        try:
                                            self._preview_cache[key] = low
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Attempt Qt-native FBO first
                    if getattr(self, 'qt_fbo_renderer', None) is not None:
                        try:
                            preview_img = self.qt_fbo_renderer.render({'verts': verts, 'faces': faces, 'colors': colors}, size=(self.preview_label.width() or 640, self.preview_label.height() or 480))
                            print('[TIN] preview generated by OffscreenQtFBORenderer')
                        except Exception as e:
                            print('[TIN] OffscreenQtFBORenderer failed:', e)

                    # Removed legacy PyOpenGL and persistent hidden GL fallbacks.
                    # If qt_fbo_renderer produced None or is unavailable, we'll fall back to saving a matplotlib preview below.

                    # If we have an image, display and save it
                    if preview_img is not None:
                        try:
                            from PyQt5.QtGui import QPixmap
                            pix = QPixmap.fromImage(preview_img)
                            if getattr(self, 'preview_label', None) is not None:
                                try:
                                    self.preview_label.setPixmap(pix.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                                except Exception:
                                    self.preview_label.setPixmap(pix)
                            # save preview images
                            try:
                                out_dir = getattr(self.sim, 'model_dir', None) or '.'
                                out_path = os.path.join(out_dir, 'outputs', 'tin_preview.png')
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                preview_img.save(out_path)
                                print(f'[TIN] preview saved: {out_path}')
                            except Exception:
                                pass
                            try:
                                from datetime import datetime
                                repo_out_dir = os.path.join(os.getcwd(), 'outputs')
                                os.makedirs(repo_out_dir, exist_ok=True)
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                repo_path = os.path.join(repo_out_dir, f'tin_preview_{ts}.png')
                                preview_img.save(repo_path)
                                print(f'[TIN] repo preview saved: {repo_path}')
                            except Exception:
                                pass
                        except Exception as e:
                            print('[TIN] failed to display/save preview image:', e)
                    else:
                        # Last-resort: matplotlib preview
                        try:
                            import matplotlib.pyplot as plt
                            from io import BytesIO
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.triplot(verts[:, 0], verts[:, 1], faces, linewidth=0.5)
                            ax.set_aspect('equal')
                            ax.axis('off')
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                            plt.close(fig)
                            buf.seek(0)
                            from PyQt5.QtGui import QImage, QPixmap
                            qim = QImage.fromData(buf.getvalue())
                            pix = QPixmap.fromImage(qim)
                            if getattr(self, 'preview_label', None) is not None:
                                try:
                                    self.preview_label.setPixmap(pix.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio))
                                except Exception:
                                    self.preview_label.setPixmap(pix)
                            # save matplotlib preview
                            try:
                                out_dir = getattr(self.sim, 'model_dir', None) or '.'
                                out_path = os.path.join(out_dir, 'outputs', 'tin_preview.png')
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with open(out_path, 'wb') as f:
                                    f.write(buf.getvalue())
                                print(f'[TIN] preview saved: {out_path}')
                            except Exception:
                                pass
                        except Exception as e:
                            print('[TIN] matplotlib preview failed:', e)
                except Exception as e:
                    print('[TIN] preview generation failed:', e)
                    # save to disk when no GL view available
                    try:
                        if out_path is not None:
                            with open(out_path, 'wb') as f:
                                f.write(buf.getvalue())
                            print(f'[TIN] preview saved: {out_path}')
                            try:
                                repo_out_dir = os.path.join(os.getcwd(), 'outputs')
                                os.makedirs(repo_out_dir, exist_ok=True)
                                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                repo_path = os.path.join(repo_out_dir, f'tin_preview_{ts}.png')
                                with open(repo_path, 'wb') as f:
                                    f.write(buf.getvalue())
                                print(f'[TIN] repo preview saved: {repo_path}')
                            except Exception:
                                pass
                    except Exception:
                        pass

            builder.mesh_ready.connect(_on_mesh)
            try:
                builder.start()
            except Exception:
                builder.run()
        except Exception as e:
            print('[RENDER] setup_background failed:', e)

    def rebuild_tin_action(self):
        try:
            if hasattr(self, 've_slider'):
                self.sim.vert_exag = self.ve_slider.value() / 100.0
            QtCore.QTimer.singleShot(10, self.setup_background)
        except Exception as e:
            print('[REBUILD] Exception rebuilding TIN:', e)

    def update_simulation(self):
        if self.paused:
            return
        try:
            # Require the simulation to provide a PID controller. No fallback.
            pid = getattr(self.sim, 'pid_controller', None)
            if pid is None:
                raise RuntimeError('Simulation has no pid_controller; ensure the sim creates it before visualization (no fallback allowed)')

            self.sim.timestep(self.current_timestep, self.dt, 9.81, pid)
            self.current_timestep += 1
            # RL training logic
            if self.rl_trainer:
                try:
                    self.update_rl_training()
                except Exception as e:
                    print('[ERROR] update_rl_training failed:', e)
            # Update displays
            self.update_displays()
        except Exception as e:
            print('[ERROR] simulation step failed:', e)
            self.paused = True

    def update_rl_training(self):
        """Update RL training metrics and episode management (ported behavior)."""
        # Extract current state
        try:
            current_metrics = self.rl_trainer.extract_state_metrics()
        except Exception:
            current_metrics = {}

        try:
            self.update_metrics_panel(current_metrics)
        except Exception:
            pass

        # Compute reward
        try:
            if self.prev_metrics is not None:
                reward = self.rl_trainer.compute_reward(self.prev_metrics, current_metrics)
                self.episode_reward += reward
        except Exception:
            pass

        self.prev_metrics = current_metrics

        # Check if episode complete
        if self.current_timestep >= self.n_timesteps:
            print('\n' + ('='*80))
            print(f'EPISODE {self.current_episode} COMPLETE | Reward: {self.episode_reward:.2f}')
            self.rewards_history.append(self.episode_reward)

            # Update best and save weights
            try:
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward
                    print(f'*** NEW BEST REWARD: {self.best_reward:.2f}')
                    import json, os
                    save_dir = os.path.join(os.getcwd(), 'outputs', 'rl_training')
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'best_weights.json')
                    with open(save_path, 'w') as f:
                        json.dump(self.rl_trainer.behavioral_weights.to_dict(), f, indent=2)
                    print(f'Saved to {save_path}')
            except Exception as e:
                print('[ERROR] Saving best weights failed:', e)

            # Mutate weights for next episode
            try:
                self.rl_trainer.behavioral_weights.mutate(scale=0.1)
                self.sim.apply_behavioral_weights(self.rl_trainer.behavioral_weights)
            except Exception as e:
                print('[ERROR] Mutating/applying weights failed:', e)

            # Reset for next episode (including agent positions)
            try:
                self.sim.reset_spatial_state(reset_positions=True)
            except Exception:
                try:
                    self.sim.reset_spatial_state()
                except Exception:
                    pass
            self.current_episode += 1
            self.current_timestep = 0
            self.episode_reward = 0.0
            self.prev_metrics = None

            # Update reward plot
            try:
                if hasattr(self, 'reward_plot') and self.reward_plot is not None:
                    from pyqtgraph import mkPen
                    self.reward_plot.plot(range(len(self.rewards_history)), self.rewards_history, pen=mkPen('g', width=2), clear=True)
            except Exception:
                pass

            # Compute and append per-episode means for any tracked metrics
            try:
                if hasattr(self, 'episode_metric_accumulators'):
                    for m, vals in self.episode_metric_accumulators.items():
                        if len(vals) == 0:
                            continue
                        mean_val = float(np.mean(vals))
                        if m not in self.per_episode_series:
                            self.per_episode_series[m] = []
                        self.per_episode_series[m].append(mean_val)
                        if m not in self.per_episode_handles:
                            pen = pg.mkPen('y', width=2)
                            self.per_episode_handles[m] = self.per_episode_plot.plot(list(range(len(self.per_episode_series[m]))), self.per_episode_series[m], pen=pen, name=m)
                        else:
                            handle = self.per_episode_handles[m]
                            handle.setData(list(range(len(self.per_episode_series[m]))), self.per_episode_series[m])
                    # clear accumulators for next episode
                    self.episode_metric_accumulators = {}
            except Exception as e:
                print('[ERROR] Exception in per-episode metrics update:', e)

    def update_displays(self):
        # If GL is unavailable, we'll draw into the 2D `plot_widget` instead.
        use_2d = (gl is None) or (getattr(self, 'gl_view', None) is None)
        if not (hasattr(self.sim, 'X') and hasattr(self.sim, 'Y')):
            return
        num_agents = getattr(self.sim, 'num_agents', 0)
        if num_agents == 0:
            return

        dead_mask = (getattr(self.sim, 'dead', np.zeros(num_agents)) != 0)
        alive_mask = ~dead_mask

        # Respect show-dead checkbox if present
        try:
            if getattr(self, 'show_dead_cb', None) and self.show_dead_cb.isChecked():
                x = self.sim.X; y = self.sim.Y
                # color alive vs dead
                colors = np.where(alive_mask[:, None], [1.0, 0.4, 0.4, 0.9], [0.4, 0.4, 0.4, 0.4])
            else:
                x = self.sim.X[alive_mask]; y = self.sim.Y[alive_mask]
                colors = np.tile([1.0, 0.4, 0.4, 0.9], (len(x), 1))
        except Exception:
            x = self.sim.X[alive_mask]; y = self.sim.Y[alive_mask]
            colors = np.tile([1.0, 0.4, 0.4, 0.9], (len(x), 1))

        if use_2d:
            try:
                self.plot_widget.clear()
                self.plot_widget.plot(x, y, pen=None, symbol='o', symbolBrush=pg.mkBrush(255, 102, 102), symbolSize=6)
                # perimeter overlay
                if getattr(self, 'perimeter_pts', None) is not None:
                    perim = self.perimeter_pts
                    self.plot_widget.plot(perim[:, 0], perim[:, 1], pen=pg.mkPen('g', width=1))
            except Exception:
                pass
        else:
            pts = np.column_stack([x, y, np.zeros_like(x)])
            scatter = gl.GLScatterPlotItem(pos=pts, color=colors, size=6)
            try:
                if hasattr(self, 'gl_agent_scatter') and self.gl_agent_scatter is not None:
                    try:
                        self._safe_remove(self.gl_agent_scatter)
                    except Exception:
                        pass
            except Exception:
                pass
            self.gl_agent_scatter = scatter
            self._safe_add(self.gl_agent_scatter, mark='agents')

        # Direction indicators (basic implementation)
        try:
            if getattr(self, 'show_direction_cb', None) and self.show_direction_cb.isChecked() and hasattr(self.sim, 'heading'):
                if hasattr(self, 'gl_direction_lines'):
                    for item in getattr(self, 'gl_direction_lines', []):
                            try:
                                self._safe_remove(item)
                            except Exception:
                                pass
                self.gl_direction_lines = []
                if getattr(self, 'show_dead_cb', None) and self.show_dead_cb.isChecked():
                    headings = self.sim.heading
                    pos_x, pos_y = self.sim.X, self.sim.Y
                else:
                    headings = self.sim.heading[alive_mask]
                    pos_x, pos_y = self.sim.X[alive_mask], self.sim.Y[alive_mask]
                arrow_length = 5.0
                end_x = pos_x - arrow_length * np.cos(headings)
                end_y = pos_y - arrow_length * np.sin(headings)
                for i in range(len(pos_x)):
                    seg_pts = np.array([[pos_x[i], pos_y[i], 0], [end_x[i], end_y[i], 0]])
                    seg = gl.GLLinePlotItem(pos=seg_pts, color=(1.0, 0.78, 0.39, 0.6), width=1.5, antialias=True, mode='lines')
                    try:
                        self._safe_add(seg, mark='dir')
                    except Exception:
                        pass
                    self.gl_direction_lines.append(seg)
        except Exception:
            pass
        # if perimeter data exists and GL available, add perimeter overlay
        try:
            if getattr(self, 'perimeter_pts', None) is not None:
                perim = self.perimeter_pts
                if use_2d:
                    # already drawn above in plot_widget
                    pass
                else:
                    perim_pts3 = np.column_stack([perim[:,0], perim[:,1], np.zeros(len(perim))])
                    self.perim_scatter = gl.GLScatterPlotItem(pos=perim_pts3, color=(0.0, 0.8, 0.0, 1.0), size=2)
                    try:
                        self._safe_add(self.perim_scatter, mark='perim')
                    except Exception:
                        pass
        except Exception:
            pass

    def toggle_pause(self):
        self.paused = not self.paused
        try:
            if self.paused:
                self.play_btn.setEnabled(True)
                self.pause_btn.setText('Pause')
            else:
                self.play_btn.setEnabled(False)
                self.pause_btn.setText('Running')
        except Exception:
            pass

    def update_metrics_panel(self, metrics):
        """Update metrics panel labels (ported from original viewer)."""
        def fmt_metric(val):
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val) if val is not None else "--"
        try:
            self.mean_speed_label.setText(f"Mean Speed: {fmt_metric(metrics.get('mean_speed', None))}")
            self.max_speed_label.setText(f"Max Speed: {fmt_metric(metrics.get('max_speed', None))}")
            self.mean_energy_label.setText(f"Mean Energy: {fmt_metric(metrics.get('mean_energy', None))}")
            self.min_energy_label.setText(f"Min Energy: {fmt_metric(metrics.get('min_energy', None))}")
            upstream_val = metrics.get('upstream_progress', metrics.get('mean_upstream_progress', '--'))
            if isinstance(upstream_val, (int, float)):
                self.upstream_progress_label.setText(f"Upstream Progress: {upstream_val:.2f}")
            else:
                self.upstream_progress_label.setText(f"Upstream Progress: --")
            self.mean_centerline_label.setText(f"Mean Centerline: {fmt_metric(metrics.get('mean_centerline', None))}")
            self.mean_passage_delay_label.setText(f"Mean Passage Delay: {fmt_metric(metrics.get('mean_passage_delay', None))}")
            if 'passage_success_rate' in metrics:
                self.passage_success_rate_label.setText(f"Passage Success: {metrics['passage_success_rate']:.1%}")
            elif 'success_rate' in metrics:
                self.passage_success_rate_label.setText(f"Passage Success: {metrics['success_rate']:.1%}")
            self.mean_nn_dist_label.setText(f"Mean NN Dist: {fmt_metric(metrics.get('mean_nn_dist', None))}")
            self.polarization_label.setText(f"Polarization: {fmt_metric(metrics.get('polarization', None))}")

            # accumulate per-timestep values into episode accumulators if checkboxes enabled
            for m, cb in getattr(self, 'track_metric_cbs', {}).items():
                try:
                    if cb.isChecked() and m in metrics:
                        if not hasattr(self, 'episode_metric_accumulators'):
                            self.episode_metric_accumulators = {}
                        if m not in self.episode_metric_accumulators:
                            self.episode_metric_accumulators[m] = []
                        self.episode_metric_accumulators[m].append(float(metrics[m]))
                except Exception:
                    pass
        except Exception:
            pass

    def refresh_rl_labels(self):
        try:
            self.episode_label.setText(f"Episode: {getattr(self, 'current_episode', 0)} | Timestep: {getattr(self, 'current_timestep', 0)}")
            self.reward_label.setText(f"Reward: {getattr(self, 'episode_reward', 0.0):.2f}")
            self.best_reward_label.setText(f"Best: {getattr(self, 'best_reward', 0.0):.2f}")
        except Exception:
            pass

    def reset_simulation(self):
        try:
            if hasattr(self.sim, 'reset_spatial_state'):
                try:
                    self.sim.reset_spatial_state(reset_positions=True)
                except TypeError:
                    self.sim.reset_spatial_state()
        except Exception as e:
            print('[RESET] Exception:', e)
        self.current_timestep = 0
        self.paused = True

    def run(self):
        self.show()
        try:
            # attempt to raise and activate the window so it appears on top
            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass
            print('[VIEWER] shown')
        except Exception:
            pass
        return QtWidgets.QApplication.instance().exec_()


def launch_viewer(simulation, dt=0.1, T=600, rl_trainer=None, **kwargs):
    try:
        print('[LAUNCH] checking for existing QApplication')
        app = QtWidgets.QApplication.instance()
        if app is None:
            print('[LAUNCH] creating QApplication')
            try:
                # Prefer desktop OpenGL at the application level; must be set
                # before QApplication is constructed.
                try:
                    from PyQt5.QtCore import QCoreApplication
                    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
                    print('[GL] QCoreApplication attribute AA_UseDesktopOpenGL set')
                except Exception as e:
                    print('[GL] Failed to set AA_UseDesktopOpenGL attribute:', e)
            except Exception:
                pass
            # Set a reasonable default surface format to enable desktop GL
            try:
                fmt = QSurfaceFormat()
                fmt.setRenderableType(QSurfaceFormat.OpenGL)
                try:
                    # Prefer a recent Core profile (desktop GL) which is
                    # likely to provide modern GL functions.
                    fmt.setProfile(QSurfaceFormat.CoreProfile)
                    fmt.setVersion(3, 3)
                    fmt.setDepthBufferSize(24)
                    QSurfaceFormat.setDefaultFormat(fmt)
                    print('[GLFMT] QSurfaceFormat set: OpenGL 3.3 Core')
                except Exception:
                    try:
                        # Fallback to compatibility 2.0 for legacy drivers
                        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
                        fmt.setVersion(2, 0)
                        QSurfaceFormat.setDefaultFormat(fmt)
                        print('[GLFMT] QSurfaceFormat set: OpenGL 2.0 Compatibility (fallback)')
                    except Exception:
                        try:
                            # Final fallback: Core 3.2
                            fmt.setProfile(QSurfaceFormat.CoreProfile)
                            fmt.setVersion(3, 2)
                            QSurfaceFormat.setDefaultFormat(fmt)
                            print('[GLFMT] QSurfaceFormat set: OpenGL 3.2 Core (final fallback)')
                        except Exception as e:
                            print('[GLFMT] Failed to set requested QSurfaceFormat profiles:', e)
            except Exception as e:
                print('[GLFMT] Failed to set QSurfaceFormat:', e)
            app = QtWidgets.QApplication(sys.argv)
        viewer = SalmonViewer(simulation, dt=dt, T=T, rl_trainer=rl_trainer, **kwargs)
        print('[LAUNCH] launching viewer.run()')
        rc = viewer.run()
        print(f'[LAUNCH] viewer.run() exited with code {rc}')
        return rc
    except Exception as e:
        print('[LAUNCH] Exception launching viewer:', e)
        import traceback
        traceback.print_exc()
        raise
