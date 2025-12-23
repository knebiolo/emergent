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
    from PyQt5.QtCore import QTimer, Qt
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

        xs = positions[:, :, 0]
        ys = positions[:, :, 1]
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

    def paintGL(self):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(255, 255, 255))

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
        pen = QPen(QColor(180, 10, 10))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        brush = QColor(220, 30, 30)
        painter.setBrush(brush)
        r = max(1, int(min(w, h) * 0.002))
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


class MainWindow(QMainWindow):
    def __init__(self, positions: np.ndarray):
        super().__init__()
        self.setWindowTitle("Realtime Simulation Viewer")
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

    def _on_speed(self, v: int):
        mult = v / 100.0
        self.viewer.set_speed(mult if mult > 0 else 1.0)


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
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)

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

    win = MainWindow(positions)
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
