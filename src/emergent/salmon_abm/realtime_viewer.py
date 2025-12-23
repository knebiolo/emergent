#!/usr/bin/env python3
"""
Realtime simulation viewer (Qt) for emergent salmon_abm simulations.

Place this file in the package `src/emergent/salmon_abm` so it is part of
the library distribution and can be imported as `emergent.salmon_abm.realtime_viewer`.

The viewer loads an HDF5 (.h5/.hdf5) or CSV trace produced by the runner and
plays back agent positions with Start/Pause/Stop/Restart and a speed slider.

This module intentionally avoids complex OpenGL dependencies and uses
`QOpenGLWidget` together with `QPainter` for simple, fast 2D rendering.

Requirements (runtime): PyQt5, numpy, h5py (for HDF5 playback)

Run:
  python -m emergent.salmon_abm.realtime_viewer path/to/sim_db.h5

"""
from __future__ import annotations

import sys
import os
import argparse
from typing import Optional

import numpy as np

try:
    import h5py
except Exception:
    h5py = None

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QPushButton,
        QSlider,
        QFileDialog,
        QLabel,
        QHBoxLayout,
        QVBoxLayout,
        QOpenGLWidget,
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread
    from PyQt5.QtGui import QPainter, QColor, QPen
except Exception:
    raise


def load_positions_from_h5(path: str) -> np.ndarray:
    """Heuristic loader for HDF5 position datasets.

    Returns an array shaped (T, N, 2) of floats. Raises on failure.
    """
    if h5py is None:
        raise RuntimeError("h5py is required to load HDF5 files")
    f = h5py.File(path, "r")

    datasets = {}

    def collect(group, prefix=""):
        for k, v in group.items():
            name = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                datasets[name] = v
            else:
                collect(v, name)

    collect(f)

    # Common pattern: separate x and y coordinate datasets
    x_ds = None
    y_ds = None
    for name in datasets:
        ln = name.lower()
        if ln.endswith("/x") or ln.endswith("_x") or ln.endswith("/x_coords") or ln == "x_coords":
            x_ds = datasets[name]
        if ln.endswith("/y") or ln.endswith("_y") or ln.endswith("/y_coords") or ln == "y_coords":
            y_ds = datasets[name]

    if x_ds is not None and y_ds is not None and x_ds.shape == y_ds.shape:
        X = np.array(x_ds)
        Y = np.array(y_ds)
        f.close()
        if X.ndim == 2 and Y.ndim == 2:
            # assume (T, N)
            return np.stack((X, Y), axis=2)
        raise RuntimeError(f"Unsupported x/y shapes: {X.shape} / {Y.shape}")

    # Look for a combined positions dataset
    for candidate in ("positions", "position", "agents/positions", "agents/position"):
        if candidate in datasets:
            d = datasets[candidate]
            arr = np.array(d)
            f.close()
            if arr.ndim == 3 and arr.shape[2] == 2:
                # normalize to (T, N, 2)
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.transpose(1, 0, 2)
                return arr

    # fallback: any 3D dataset with last dim 2
    for name, ds in datasets.items():
        shp = ds.shape
        if len(shp) == 3 and shp[2] == 2:
            arr = np.array(ds)
            if arr.shape[0] < arr.shape[1]:
                arr = arr.transpose(1, 0, 2)
            f.close()
            return arr

    f.close()
    raise RuntimeError("Could not find positions in HDF5 file; available datasets: " + ",".join(datasets.keys()))


def load_positions_from_csv(path: str) -> np.ndarray:
    import csv

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError("Empty CSV")
    cols = [c.lower() for c in rows[0].keys()]
    has_t = any(c in ("t", "time") for c in cols)
    has_agent = any(c in ("agent", "id") for c in cols)
    has_x = any(c in ("x", "lon", "longitude") for c in cols)
    has_y = any(c in ("y", "lat", "latitude") for c in cols)
    if has_t and has_agent and has_x and has_y:
        times = sorted({float(r[[k for k in r.keys() if k.lower() in ("t", "time")][0]]) for r in rows})
        agents = sorted({int(r[[k for k in r.keys() if k.lower() in ("agent", "id")][0]]) for r in rows})
        T = len(times)
        N = len(agents)
        idx_time = {t: i for i, t in enumerate(times)}
        idx_agent = {a: i for i, a in enumerate(agents)}
        arr = np.full((T, N, 2), np.nan, dtype=float)
        for r in rows:
            t = float(r[[k for k in r.keys() if k.lower() in ("t", "time")][0]])
            a = int(r[[k for k in r.keys() if k.lower() in ("agent", "id")][0]])
            x = float(r[[k for k in r.keys() if k.lower() in ("x", "lon", "longitude")][0]])
            y = float(r[[k for k in r.keys() if k.lower() in ("y", "lat", "latitude")][0]])
            arr[idx_time[t], idx_agent[a], 0] = x
            arr[idx_time[t], idx_agent[a], 1] = y
        return arr
    raise RuntimeError("CSV format not recognized: need columns time/agent/x/y")


class ReplayWidget(QOpenGLWidget):
    def __init__(self, positions: np.ndarray, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must be (T, N, 2)")
        self.positions = positions
        self.T, self.N, _ = positions.shape
        self.frame = 0
        self.playing = False
        self.trail = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.base_interval_ms = 50
        self.timer.setInterval(self.base_interval_ms)
        # live-related attributes
        self._live_thread = None
        self._live_worker = None

        # compute bounds from provided positions
        xs = self.positions[:, :, 0]
        ys = self.positions[:, :, 1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if np.any(valid):
            self.xmin = float(np.nanmin(xs[valid]))
            self.xmax = float(np.nanmax(xs[valid]))
            self.ymin = float(np.nanmin(ys[valid]))
            self.ymax = float(np.nanmax(ys[valid]))
        else:
            self.xmin = 0.0
            self.xmax = 1.0
            self.ymin = 0.0
            self.ymax = 1.0
        # debug flag to force larger, visible points when in live-diagnostic mode
        self._debug_force_big = False

    def _tick(self):
        if not getattr(self, 'playing', False):
            return
        self.frame += 1
        if self.frame >= self.T:
            self.frame = self.T - 1
            self.playing = False
            self.timer.stop()
        self.update()

    def start(self):
        if self.frame >= self.T - 1:
            self.frame = 0
        self.playing = True
        if not self.timer.isActive():
            self.timer.start()

    def pause(self):
        self.playing = False
        if self.timer.isActive():
            self.timer.stop()

    def stop(self):
        self.playing = False
        self.frame = 0
        if self.timer.isActive():
            self.timer.stop()
        self.update()

    def restart(self):
        self.frame = 0
        self.playing = True
        if not self.timer.isActive():
            self.timer.start()
        self.update()

    def set_speed(self, multiplier: float):
        interval = max(1, int(self.base_interval_ms / float(multiplier)))
        self.timer.setInterval(interval)

    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(255, 255, 255))

        # draw light grid to show canvas area
        try:
            pen = QPen(QColor(230, 230, 230))
            painter.setPen(pen)
            step = max(20, int(min(w, h) / 10))
            for x in range(0, w, step):
                painter.drawLine(x, 0, x, h)
            for y in range(0, h, step):
                painter.drawLine(0, y, w, y)
        except Exception:
            pass

        dx = self.xmax - self.xmin
        dy = self.ymax - self.ymin
        if dx == 0:
            dx = 1.0
        if dy == 0:
            dy = 1.0
        sx = w / dx
        sy = h / dy
        s = min(sx, sy) * 0.95
        tx = (w - s * dx) / 2.0
        ty = (h - s * dy) / 2.0

        pts = self.positions[self.frame]
        # if no agents, show waiting message
        try:
            if self.N == 0 or pts.size == 0:
                painter.setPen(QPen(QColor(80, 80, 80)))
                painter.drawText(int(w / 2) - 80, int(h / 2), 'Waiting for frames...')
                painter.end()
                return
        except Exception:
            pass
        pen = QPen(QColor(180, 10, 10))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        brush = QColor(220, 30, 30)
        painter.setBrush(brush)
        r = max(1, int(min(w, h) * 0.002))
        if getattr(self, '_debug_force_big', False):
            # force large visible dots for live debugging
            r = max(r, int(min(w, h) * 0.01))
        for i in range(self.N):
            x, y = pts[i]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            sxp = tx + (x - self.xmin) * s
            syp = ty + (self.ymax - y) * s
            painter.drawEllipse(int(sxp) - r, int(syp) - r, 2 * r, 2 * r)

        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(6, 14, f"Frame: {self.frame+1}/{self.T}  Agents: {self.N}")
        painter.end()


class GLViewer(QOpenGLWidget):
    """OpenGL VBO-backed viewer for large numbers of agents.

    This attempts to use PyOpenGL. If not available, constructing this
    widget will raise ImportError and the caller should fall back.
    """

    def __init__(self, positions: np.ndarray, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must be (T, N, 2)")
        self.positions = positions
        self.T, self.N, _ = positions.shape
        self.frame = 0
        self.playing = False
        self.base_interval_ms = 50
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(self.base_interval_ms)
        self._vbo = None
        self._program = None
        self._gl_available = False

        try:
            from OpenGL import GL
            from OpenGL.arrays import vbo as glvbo
            self._GL = GL
            self._glvbo = glvbo
            self._gl_available = True
        except Exception:
            self._gl_available = False
        # compute initial world bounds (avoid degenerate projection)
        try:
            xs = self.positions[:, :, 0]
            ys = self.positions[:, :, 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            if np.any(valid):
                self.xmin = float(np.nanmin(xs[valid]))
                self.xmax = float(np.nanmax(xs[valid]))
                self.ymin = float(np.nanmin(ys[valid]))
                self.ymax = float(np.nanmax(ys[valid]))
            else:
                self.xmin, self.xmax, self.ymin, self.ymax = 0.0, 1.0, 0.0, 1.0
        except Exception:
            self.xmin, self.xmax, self.ymin, self.ymax = 0.0, 1.0, 0.0, 1.0

    def _tick(self):
        if not self.playing:
            return
        self.frame += 1
        if self.frame >= self.T:
            self.frame = self.T - 1
            self.playing = False
            self.timer.stop()
        self.update()

    def start(self):
        if self.frame >= self.T - 1:
            self.frame = 0
        self.playing = True
        if not self.timer.isActive():
            self.timer.start()

    def pause(self):
        self.playing = False
        if self.timer.isActive():
            self.timer.stop()

    def stop(self):
        self.playing = False
        self.frame = 0
        if self.timer.isActive():
            self.timer.stop()
        self.update()

    def restart(self):
        self.frame = 0
        self.playing = True
        if not self.timer.isActive():
            self.timer.start()
        self.update()

    def set_speed(self, multiplier: float):
        interval = max(1, int(self.base_interval_ms / float(multiplier)))
        self.timer.setInterval(interval)

    def initializeGL(self):
        if not self._gl_available:
            return
        GL = self._GL
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        try:
            # create an empty VBO
            self._vbo = self._glvbo.VBO(np.zeros((0, 2), dtype=np.float32))
        except Exception:
            self._vbo = None

    def paintGL(self):
        if not self._gl_available:
            # fallback to software painter
            painter = QPainter(self)
            painter.drawText(10, 20, 'OpenGL not available')
            painter.end()
            return
        GL = self._GL
        # ensure viewport and clear
        try:
            GL.glViewport(0, 0, int(self.width()), int(self.height()))
        except Exception:
            pass
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # fetch current positions and draw as points
        try:
            pts = self.positions[self.frame].astype(np.float32)
            # set projection to world coordinates
            try:
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                # Note: OpenGL bottom-left origin; flip Y by swapping ymin/ymax for correct orientation
                GL.glOrtho(self.xmin, self.xmax, self.ymax, self.ymin, -1.0, 1.0)
                GL.glMatrixMode(GL.GL_MODELVIEW)
                GL.glLoadIdentity()
            except Exception:
                pass
            if self._vbo is None:
                # fallback to immediate mode
                try:
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').debug('GLViewer.paintGL immediate mode: pts=%s bounds=(%s,%s,%s,%s)',
                                                              getattr(pts, 'shape', None), self.xmin, self.xmax, self.ymin, self.ymax)
                except Exception:
                    pass
                GL.glColor3f(0.8, 0.12, 0.12)
                # draw larger points so they're visible
                GL.glPointSize(6.0)
                GL.glBegin(GL.GL_POINTS)
                for x, y in pts:
                    try:
                        GL.glVertex2f(float(x), float(y))
                    except Exception:
                        pass
                GL.glEnd()
                # draw a visible cross at world center for debugging
                try:
                    cx = 0.5 * (self.xmin + self.xmax)
                    cy = 0.5 * (self.ymin + self.ymax)
                    GL.glColor3f(0.0, 0.0, 0.0)
                    GL.glLineWidth(2.0)
                    GL.glBegin(GL.GL_LINES)
                    GL.glVertex2f(cx - (self.xmax - self.xmin) * 0.01, cy)
                    GL.glVertex2f(cx + (self.xmax - self.xmin) * 0.01, cy)
                    GL.glVertex2f(cx, cy - (self.ymax - self.ymin) * 0.01)
                    GL.glVertex2f(cx, cy + (self.ymax - self.ymin) * 0.01)
                    GL.glEnd()
                except Exception:
                    pass
            else:
                # update VBO with current points and draw with vertex arrays
                try:
                    self._vbo.set_array(pts)
                    self._vbo.bind()
                    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                    GL.glVertexPointer(2, GL.GL_FLOAT, 0, self._vbo)
                    GL.glPointSize(3.0)
                    GL.glDrawArrays(GL.GL_POINTS, 0, pts.shape[0])
                    GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
                    self._vbo.unbind()
                except Exception:
                    # final fallback to immediate mode
                    GL.glColor3f(0.8, 0.12, 0.12)
                    GL.glPointSize(3.0)
                    GL.glBegin(GL.GL_POINTS)
                    for x, y in pts:
                        GL.glVertex2f(float(x), float(y))
                    GL.glEnd()
        except Exception:
            pass

    def update_frame(self, arr):
        """Called from GUI thread with an (N,2) float array to update current points."""
        try:
            pts = np.asarray(arr, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return
            # update world bounds
            xs = pts[:, 0]
            ys = pts[:, 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            if np.any(valid):
                self.xmin = float(np.nanmin(xs[valid]))
                self.xmax = float(np.nanmax(xs[valid]))
                self.ymin = float(np.nanmin(ys[valid]))
                self.ymax = float(np.nanmax(ys[valid]))
            # assign into positions buffer so paintGL can access indexed by frame
            try:
                self.positions = pts[np.newaxis, :, :]
                self.T, self.N = self.positions.shape[0], self.positions.shape[1]
            except Exception:
                pass
            # if VBO present, update it now
            if self._gl_available and self._vbo is not None:
                try:
                    self._vbo.set_array(pts)
                except Exception:
                    pass
            self.update()
        except Exception:
            pass


    def shutdown_live(self):
        try:
            if self._live_worker is not None:
                try:
                    self._live_worker.stop()
                except Exception:
                    pass
            if self._live_thread is not None:
                try:
                    # ask thread to quit and wait a short while
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').debug('Requesting live thread quit')
                    self._live_thread.quit()
                    # wait up to 2s for clean stop
                    if not self._live_thread.wait(2000):
                        _lg.getLogger('realtime_viewer').warning('Live thread did not stop after quit(); terminating')
                        try:
                            self._live_thread.terminate()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self, positions: np.ndarray, watchdog_seconds: float = 5.0):
        super().__init__()
        self.setWindowTitle("Realtime Simulation Viewer")
        self._watchdog_seconds = float(watchdog_seconds)
        self.viewer = ReplayWidget(positions)

        btn_start = QPushButton("Start")
        btn_pause = QPushButton("Pause")
        btn_stop = QPushButton("Stop")
        btn_restart = QPushButton("Restart")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 400)
        self.speed_slider.setValue(100)
        lbl_speed = QLabel("Speed")

        btn_start.clicked.connect(self.viewer.start)
        btn_pause.clicked.connect(self.viewer.pause)
        btn_stop.clicked.connect(self.viewer.stop)
        btn_restart.clicked.connect(self.viewer.restart)
        self.speed_slider.valueChanged.connect(self._on_speed)

        hl = QHBoxLayout()
        hl.addWidget(btn_start)
        hl.addWidget(btn_pause)
        hl.addWidget(btn_stop)
        hl.addWidget(btn_restart)
        hl.addWidget(lbl_speed)
        hl.addWidget(self.speed_slider)

        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addLayout(hl)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def closeEvent(self, event):
        # ensure viewer widget cleans up any live sockets/timers
        try:
            import logging as _lg
            _lg.getLogger('realtime_viewer').debug('MainWindow.closeEvent: initiating shutdown_live')
            self.viewer.shutdown_live()
        except Exception:
            pass

        # if there is still a live thread, start a short watchdog to force-stop
        try:
            thr = getattr(self.viewer, '_live_thread', None)
            worker = getattr(self.viewer, '_live_worker', None)
            if thr is not None and thr.isRunning():
                import logging as _lg
                log = _lg.getLogger('realtime_viewer')
                # request worker stop if available
                try:
                    if worker is not None:
                        worker.stop()
                except Exception:
                    pass
                # give it some time to stop gracefully
                waited = 0.0
                step = 0.1
                max_wait = max(0.1, float(self._watchdog_seconds))
                log.debug('Waiting up to %.2fs for live thread to stop', max_wait)
                while thr.isRunning() and waited < max_wait:
                    QThread.msleep(int(step * 1000))
                    waited += step
                if thr.isRunning():
                    try:
                        log.debug('Requesting thread quit()')
                        thr.quit()
                        if not thr.wait(int(max_wait * 1000)):
                            log.warning('Thread did not stop; terminating')
                            try:
                                thr.terminate()
                            except Exception:
                                pass
                    except Exception:
                        try:
                            thr.terminate()
                        except Exception:
                            pass
        except Exception:
            pass

        return super().closeEvent(event)

    def _on_speed(self, v: int):
        mult = v / 100.0
        self.viewer.set_speed(mult if mult > 0 else 1.0)


class LiveReceiver(QObject):
    """Worker that connects to a TCP server and emits numpy frames."""
    frame_received = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self._running = False
        self._sock = None

    def stop(self):
        self._running = False
        try:
            if self._sock is not None:
                try:
                    self._sock.shutdown(2)
                except Exception:
                    pass
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        except Exception:
            pass

    def run(self):
        import socket, io, struct, time

        self._running = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # retry connect a few times
        connected = False
        for _ in range(60):
            try:
                sock.connect((self.host, self.port))
                connected = True
                break
            except Exception:
                time.sleep(0.2)
        if not connected:
            try:
                self.error.emit(f'connect_failed:{self.host}:{self.port}')
            except Exception:
                pass
            return
        sock.setblocking(True)
        self._sock = sock
        try:
            while self._running:
                first = sock.recv(1)
                if not first:
                    break
                if not first:
                    break
                # If first byte is 'R' -> raw protocol: read next 4-byte length
                if first == b'R':
                    lb = sock.recv(4)
                    if not lb or len(lb) < 4:
                        self.error.emit('incomplete_raw_length')
                        break
                    if not lb or len(lb) < 4:
                        try:
                            self.error.emit('incomplete_raw_length')
                            try:
                                import logging as _lg
                                _lg.getLogger('realtime_viewer').debug('incomplete_raw_length lb=%r', lb)
                            except Exception:
                                pass
                        except Exception:
                            pass
                        break
                    (nbytes,) = struct.unpack('!I', lb)
                    try:
                        import logging as _lg
                        _lg.getLogger('realtime_viewer').debug('raw header nbytes=%d', nbytes)
                    except Exception:
                        pass
                    buf = bytearray()
                    while len(buf) < nbytes:
                        chunk = sock.recv(nbytes - len(buf))
                        if not chunk:
                            break
                        buf.extend(chunk)
                    if len(buf) < nbytes:
                        try:
                            self.error.emit('incomplete_raw_payload')
                            try:
                                import logging as _lg
                                _lg.getLogger('realtime_viewer').debug('incomplete_raw_payload expected=%d got=%d', nbytes, len(buf))
                            except Exception:
                                pass
                        except Exception:
                            pass
                        continue
                    arr = np.frombuffer(bytes(buf), dtype=np.float32)
                    if arr.size % 2 != 0:
                        self.error.emit('raw_payload_not_even')
                        continue
                    arr = arr.reshape((-1, 2))
                else:
                    # first is first byte of 4-byte big-endian length for numpy case
                    rest = sock.recv(3)
                    if not rest or len(rest) < 3:
                        self.error.emit('incomplete_length')
                        break
                    length_bytes = first + rest
                    (nbytes,) = struct.unpack('!I', length_bytes)
                    try:
                        import logging as _lg
                        _lg.getLogger('realtime_viewer').debug('npy header length=%d', nbytes)
                    except Exception:
                        pass
                    buf = bytearray()
                    while len(buf) < nbytes:
                        chunk = sock.recv(nbytes - len(buf))
                        if not chunk:
                            break
                        buf.extend(chunk)
                    bio = io.BytesIO(bytes(buf))
                    try:
                        arr = np.load(bio)
                    except Exception as e:
                        try:
                            self.error.emit(f'npy_load_error:{e}')
                        except Exception:
                            pass
                        continue
                # emit to GUI thread
                try:
                    try:
                        # also write a minimal trace file for robust diagnostics
                        with open('viewer_frame_trace.txt', 'a') as _tf:
                            _tf.write(f'frame_received shape={getattr(arr, "shape", None)}\n')
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    self.frame_received.emit(arr)
                except Exception as e:
                    try:
                        self.error.emit(f'emit_error:{e}')
                    except Exception:
                        pass
        except Exception as e:
            try:
                self.error.emit(f'worker_exception:{e}')
            except Exception:
                pass
        finally:
            try:
                sock.close()
            except Exception:
                pass


def load_any(path: str) -> np.ndarray:
    path = os.path.abspath(path)
    if path.lower().endswith(('.h5', '.hdf5')):
        return load_positions_from_h5(path)
    if path.lower().endswith('.csv'):
        return load_positions_from_csv(path)
    if h5py is not None:
        try:
            return load_positions_from_h5(path)
        except Exception:
            pass
    try:
        return load_positions_from_csv(path)
    except Exception:
        pass
    raise RuntimeError("Unrecognized file or failed to load positions: " + path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="HDF5 (.h5) or CSV file with simulation positions")
    parser.add_argument("--live", dest="live", action="store_true", help="Connect to a live simulation TCP stream instead of loading a file")
    parser.add_argument("--host", dest="host", default="127.0.0.1", help="Host for live stream (default 127.0.0.1)")
    parser.add_argument("--port", dest="port", type=int, default=50007, help="Port for live stream (default 50007)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable verbose debug logging to viewer_debug.log and stderr")
    parser.add_argument("--use-pg", dest="use_pg", action="store_true", help="Use pyqtgraph ScatterPlotItem for rendering (faster for many agents)")
    parser.add_argument("--use-gl", dest="use_gl", action="store_true", help="Use OpenGL VBO renderer for very large agent counts (best performance if PyOpenGL available)")
    parser.add_argument("--watchdog-seconds", dest="watchdog_seconds", type=float, default=5.0, help="Force-stop live receiver thread after this many seconds when closing (default 5.0)")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)

    # configure logging
    import logging

    logger = logging.getLogger('realtime_viewer')
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    fh = logging.FileHandler('viewer_debug.log')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if args.debug:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # live mode: connect to TCP stream and poll frames
        host = args.host
        port = args.port
        positions = np.zeros((1, 0, 2), dtype=float)
        win = MainWindow(positions, watchdog_seconds=args.watchdog_seconds)
        # add status label for errors
        try:
            status = QLabel('Ready')
            win.statusBar = status
        except Exception:
            status = None
        win.resize(1000, 700)
        win.show()

        # Start a LiveReceiver in a QThread to receive frames without blocking the GUI
        thread = QThread()
        worker = LiveReceiver(host, port)
        worker.moveToThread(thread)

        def handle_frame(arr):
            # arr is (N,2) -> convert to (T=1,N,2)
            try:
                import logging as _lg
                log = _lg.getLogger('realtime_viewer')
                log.debug('handle_frame received arr: ndim=%s shape=%s', getattr(arr, 'ndim', None), getattr(arr, 'shape', None))
                if arr is None:
                    return
                if arr.ndim == 2 and arr.shape[1] == 2:
                    positions_live = arr[np.newaxis, :, :]
                    # If we receive a large number of agents, prefer GL or PG renderer
                    try:
                        Nagents = positions_live.shape[1]
                    except Exception:
                        Nagents = 0
                    use_gl_now = args.use_gl or (Nagents >= 1000)
                    use_pg_now = args.use_pg or (Nagents > 500 and not use_gl_now)
                    log.debug('Detected %d agents; use_gl_now=%s use_pg_now=%s', Nagents, use_gl_now, use_pg_now)
                    # For diagnosis: force the reliable software `ReplayWidget` renderer
                    # unless the user explicitly requested GL via CLI.
                    if not args.use_gl:
                        use_gl_now = False
                        use_pg_now = False
                        force_replay = True
                    else:
                        force_replay = False

                    if use_gl_now:
                        try:
                            gl_view = GLViewer(positions_live)
                            try:
                                win.centralWidget().layout().replaceWidget(win.viewer, gl_view)
                                win.viewer.setParent(None)
                                win.viewer = gl_view
                            except Exception:
                                logger.exception('Failed to swap GLViewer into MainWindow')
                        except Exception:
                            logger.exception('GLViewer init failed; falling back')
                    elif use_pg_now and not isinstance(win.viewer, QWidget):
                        # Lazily initialize PGViewer and swap it in
                        try:
                            from pyqtgraph import GraphicsLayoutWidget, ScatterPlotItem

                            class PGViewer(QWidget):
                                def __init__(self, parent=None):
                                    super().__init__(parent)
                                    self.plot = GraphicsLayoutWidget()
                                    self.sp = ScatterPlotItem(size=4, pen=None, brush=(200, 30, 30, 200))
                                    vw = self.plot.addViewBox()
                                    vw.addItem(self.sp)
                                    layout = QVBoxLayout()
                                    layout.addWidget(self.plot)
                                    self.setLayout(layout)

                                def update_frame(self, arr):
                                    try:
                                        if arr is None or arr.size == 0:
                                            self.sp.setData([])
                                            return
                                        x = arr[:, 0]
                                        y = arr[:, 1]
                                        self.sp.setData(x, y)
                                    except Exception:
                                        logger.exception('PGViewer update_frame error')

                            pg_view = PGViewer()
                            try:
                                win.centralWidget().layout().replaceWidget(win.viewer, pg_view)
                                win.viewer.setParent(None)
                                win.viewer = pg_view
                            except Exception:
                                logger.exception('Failed to swap PGViewer into MainWindow')
                        except Exception:
                            logger.exception('pyqtgraph renderer init failed')
                    # set positions on whichever viewer we have (diagnostic: prefer ReplayWidget)
                    try:
                        # Prefer the reliable software `ReplayWidget` renderer unless GL requested
                        if not args.use_gl and not isinstance(win.viewer, ReplayWidget):
                            try:
                                replay = ReplayWidget(positions_live)
                                win.centralWidget().layout().replaceWidget(win.viewer, replay)
                                win.viewer.setParent(None)
                                win.viewer = replay
                                log.debug('Swapped in ReplayWidget for live frames')
                            except Exception:
                                log.exception('Failed to swap in ReplayWidget')
                        # compute frame bounds and log them
                        try:
                            xs = positions_live[:, :, 0]
                            ys = positions_live[:, :, 1]
                            valid = np.isfinite(xs) & np.isfinite(ys)
                            if np.any(valid):
                                xmin = float(np.nanmin(xs[valid]))
                                xmax = float(np.nanmax(xs[valid]))
                                ymin = float(np.nanmin(ys[valid]))
                                ymax = float(np.nanmax(ys[valid]))
                                log.debug('live frame bounds xmin=%s xmax=%s ymin=%s ymax=%s', xmin, xmax, ymin, ymax)
                        except Exception:
                            log.exception('Failed computing live frame bounds')

                        win.viewer.positions = positions_live
                        # Extra diagnostic logging: report bounds and trigger update
                        try:
                            xs = positions_live[:, :, 0]
                            ys = positions_live[:, :, 1]
                            valid = np.isfinite(xs) & np.isfinite(ys)
                            if np.any(valid):
                                xmin = float(np.nanmin(xs[valid]))
                                xmax = float(np.nanmax(xs[valid]))
                                ymin = float(np.nanmin(ys[valid]))
                                ymax = float(np.nanmax(ys[valid]))
                                log.debug('Applying live positions to viewer: xmin=%s xmax=%s ymin=%s ymax=%s', xmin, xmax, ymin, ymax)
                        except Exception:
                            log.exception('Failed to compute live bounds for logging')
                        # diagnostic flag disabled by default; do not force large points
                        # if legacy ReplayWidget
                        if hasattr(win.viewer, 'T'):
                            win.viewer.T, win.viewer.N = positions_live.shape[0], positions_live.shape[1]
                            win.viewer.frame = 0
                            xs = positions_live[:, :, 0]
                            ys = positions_live[:, :, 1]
                            valid = np.isfinite(xs) & np.isfinite(ys)
                            if np.any(valid):
                                win.viewer.xmin = float(np.nanmin(xs[valid]))
                                win.viewer.xmax = float(np.nanmax(xs[valid]))
                                win.viewer.ymin = float(np.nanmin(ys[valid]))
                                win.viewer.ymax = float(np.nanmax(ys[valid]))
                            # snapshot logic removed (diagnostics reverted)
                            try:
                                win.viewer.update()
                                log.debug('Called win.viewer.update()')
                            except Exception:
                                log.exception('viewer.update() failed')
                        else:
                            # assume PGViewer-like
                            try:
                                win.viewer.update_frame(arr)
                            except Exception:
                                log.exception('PGViewer update_frame failed')
                    except Exception:
                        log.exception('Error setting live positions')
            except Exception:
                logger.exception('Error handling frame')
                pass

        def handle_error(msg):
            logger.error('LiveReceiver error: %s', msg)
            try:
                if status is not None:
                    status.setText(f'ERROR: {msg}')
            except Exception:
                pass

        # worker error signal
        try:
            worker.error.connect(handle_error)
        except Exception:
            pass
        worker.frame_received.connect(handle_frame)
        thread.started.connect(worker.run)
        thread.start()
        # store references for cleanup
        win.viewer._live_thread = thread
        win.viewer._live_worker = worker
        logger.info('Started LiveReceiver thread for %s:%d', host, port)

        # optionally use pyqtgraph renderer if requested
        use_pg = args.use_pg
        if use_pg:
            try:
                from pyqtgraph import PlotWidget, GraphicsLayoutWidget, ScatterPlotItem

                class PGViewer(QWidget):
                    def __init__(self, parent=None):
                        super().__init__(parent)
                        self.plot = GraphicsLayoutWidget()
                        self.sp = ScatterPlotItem(size=5, pen=None, brush=(200, 30, 30, 200))
                        vw = self.plot.addViewBox()
                        vw.addItem(self.sp)
                        layout = QVBoxLayout()
                        layout.addWidget(self.plot)
                        self.setLayout(layout)

                    def update_frame(self, arr):
                        try:
                            if arr is None or arr.size == 0:
                                self.sp.setData([])
                                return
                            x = arr[:, 0]
                            y = arr[:, 1]
                            self.sp.setData(x, y)
                        except Exception:
                            logger.exception('PGViewer update_frame error')

                pg_view = PGViewer()
                # replace the existing viewer widget in the main layout
                try:
                    win.centralWidget().layout().replaceWidget(win.viewer, pg_view)
                    win.viewer.setParent(None)
                    win.viewer = pg_view
                except Exception:
                    logger.exception('Failed to swap PGViewer into MainWindow')
            except Exception:
                logger.exception('pyqtgraph renderer init failed')

        try:
            ret = app.exec_()
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            try:
                with open('viewer_stderr.txt', 'a') as fh:
                    fh.write('Unhandled exception in viewer:\n')
                    fh.write(tb)
            except Exception:
                pass
            try:
                win.viewer.shutdown_live()
            except Exception:
                pass
            sys.exit(1)
        sys.exit(ret)

    # file mode (fallback)
    path = args.file
    if not path:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilters(["HDF5 files (*.h5 *.hdf5)", "CSV files (*.csv)", "All files (*)"])
        if dlg.exec_():
            selected = dlg.selectedFiles()
            path = selected[0]
        else:
            print("No file selected; exiting")
            return

    try:
        positions = load_any(path)
    except Exception as e:
        print("Failed to load positions:", e)
        return

    # create window and pick renderer based on agent count
    win = MainWindow(positions)
    try:
        Nagents = positions.shape[1]
    except Exception:
        Nagents = 0
    use_gl_mode = args.use_gl or (Nagents >= 1000)
    if use_gl_mode:
        try:
            glw = GLViewer(positions)
            try:
                win.centralWidget().layout().replaceWidget(win.viewer, glw)
                win.viewer.setParent(None)
                win.viewer = glw
            except Exception:
                logger.exception('Failed to swap GLViewer into MainWindow (file mode)')
        except Exception:
            logger.exception('GLViewer init failed in file mode; using fallback')

    win.resize(1000, 700)
    win.show()
    try:
        ret = app.exec_()
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        try:
            with open('viewer_stderr.txt', 'a') as fh:
                fh.write('Unhandled exception in viewer:\n')
                fh.write(tb)
        except Exception:
            pass
        try:
            win.viewer.shutdown_live()
        except Exception:
            pass
        sys.exit(1)
    sys.exit(ret)


if __name__ == '__main__':
    main()
