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
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject, QThread, QRectF, QPointF
    from PyQt5.QtGui import QPainter, QColor, QPen, QImage
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

    # Priority check: agent_data/X and agent_data/Y (headless runner format)
    if 'agent_data/X' in datasets and 'agent_data/Y' in datasets:
        X = np.array(datasets['agent_data/X']).T  # Transpose (N, T) to (T, N)
        Y = np.array(datasets['agent_data/Y']).T
        f.close()
        return np.stack((X, Y), axis=2)  # Shape: (T, N, 2)

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
            # Check if shape is (N_agents, T_timesteps) and transpose if needed
            if X.shape[0] < X.shape[1]:
                # Likely (N, T) format - transpose to (T, N)
                X = X.T
                Y = Y.T
            # Now stack along last dimension to get (T, N, 2)
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


def load_env_from_h5(path: str):
    """Load environment depth, x_coords, y_coords from HDF5 file.
    
    Returns (depth_array, x_coords, y_coords) or (None, None, None) if not found.
    """
    if h5py is None:
        return None, None, None
    try:
        f = h5py.File(path, "r")
        if 'environment/depth' in f and 'environment/x_coords' in f and 'environment/y_coords' in f:
            depth = np.array(f['environment/depth'])
            x_coords = np.array(f['environment/x_coords'])
            y_coords = np.array(f['environment/y_coords'])
            f.close()
            return depth, x_coords, y_coords
        f.close()
    except Exception:
        pass
    return None, None, None


def load_battery_from_h5(path: str):
    """Load battery state array from HDF5 file if available.
    
    Returns:
        np.ndarray or None: Battery array of shape (T, N) where T=timesteps, N=agents.
                           Values range from 0.0 (depleted) to 1.0 (full charge).
    """
    if h5py is None:
        return None
    try:
        f = h5py.File(path, "r")
        if 'agent_data/battery' in f:
            battery = np.array(f['agent_data/battery']).T  # Transpose (N, T) to (T, N)
            f.close()
            return battery
        f.close()
    except Exception:
        pass
    return None


def load_heading_from_h5(path: str):
    """Load heading array from HDF5 file if available.
    
    Returns:
        np.ndarray or None: Heading array of shape (T, N) where T=timesteps, N=agents.
                           Values are in radians.
    """
    if h5py is None:
        return None
    try:
        f = h5py.File(path, "r")
        if 'agent_data/heading' in f:
            heading = np.array(f['agent_data/heading']).T  # Transpose (N, T) to (T, N)
            f.close()
            return heading
        f.close()
    except Exception:
        pass
    return None


def load_positions_from_csv(path: str) -> np.ndarray:
    import csv

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError("Empty CSV")
    cols = [c.lower() for c in rows[0].keys()]
    has_t = any(c in ("t", "time", "timestep") for c in cols)
    has_agent = any(c in ("agent", "id") for c in cols)
    has_x = any(c in ("x", "lon", "longitude") for c in cols)
    has_y = any(c in ("y", "lat", "latitude") for c in cols)
    if has_t and has_agent and has_x and has_y:
        times = sorted({float(r[[k for k in r.keys() if k.lower() in ("t", "time", "timestep")][0]]) for r in rows})
        agents = sorted({int(r[[k for k in r.keys() if k.lower() in ("agent", "id")][0]]) for r in rows})
        T = len(times)
        N = len(agents)
        idx_time = {t: i for i, t in enumerate(times)}
        idx_agent = {a: i for i, a in enumerate(agents)}
        arr = np.full((T, N, 2), np.nan, dtype=float)
        for r in rows:
            t = float(r[[k for k in r.keys() if k.lower() in ("t", "time", "timestep")][0]])
            a = int(r[[k for k in r.keys() if k.lower() in ("agent", "id")][0]])
            x = float(r[[k for k in r.keys() if k.lower() in ("x", "lon", "longitude")][0]])
            y = float(r[[k for k in r.keys() if k.lower() in ("y", "lat", "latitude")][0]])
            arr[idx_time[t], idx_agent[a], 0] = x
            arr[idx_time[t], idx_agent[a], 1] = y
        return arr
    raise RuntimeError("CSV format not recognized: need columns time/agent/x/y")


class ReplayWidget(QOpenGLWidget):
    def __init__(self, positions: np.ndarray, parent: Optional[QWidget] = None, pad: float = 1.15, pan_x: float = 0.0, pan_y: float = 0.0, point_size: Optional[float] = None, env_depth: Optional[str] = None, allow_expand_bounds: bool = False, smooth_alpha: float = 1.0, env_clip_pct: tuple = (0.0, 100.0), env_depth_array: Optional[np.ndarray] = None, env_x_coords: Optional[np.ndarray] = None, env_y_coords: Optional[np.ndarray] = None, battery_array: Optional[np.ndarray] = None, heading_array: Optional[np.ndarray] = None):
        super().__init__(parent)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must be (T, N, 2)")
        self.positions = positions
        self.T, self.N, _ = positions.shape
        self.frame = 0
        self.playing = False
        self.trail = 0
        self.fish_body_length = 8  # length of fish body in pixels
        self.tail_segments = 3  # number of tail segments for animation
        self.battery_array = battery_array  # Battery data for fatigue visualization
        self.heading_array = heading_array  # Heading data for oriented fish rendering
        self.max_agents = None  # Maximum number of agents to display (None = show all)

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
        # view transform state
        self._pad = float(pad)
        self._pan_x = float(pan_x)
        self._pan_y = float(pan_y)
        # point size in pixels (radius). If None, compute relative to canvas size
        self._point_size = None if point_size is None else float(point_size)
        # trail buffer for live mode (list of (N,2) arrays)
        self._trail_buf = []
        self._trail_length = 8
        # simple live frame counter (increments each update_frame call)
        self._live_count = 0
        # whether live bounds have been initialized (prevent recentering)
        self._live_fixed_bounds = False
        # whether allowed to expand bounds after initial setting
        self._allow_expand_bounds = bool(allow_expand_bounds)
        # smoothing alpha for live frames (1.0 = no smoothing)
        self._smooth_alpha = float(smooth_alpha) if smooth_alpha is not None else 1.0
        self._smoothed_positions = None
        # optional background image (QImage) rendered to the world extents
        self._bg_qimage = None
        
        # Try to load depth from arrays first, then from file
        if env_depth_array is not None and env_x_coords is not None and env_y_coords is not None:
            try:
                a = np.array(env_depth_array, dtype=float)
                # detect nodata-like values
                nodata_mask = np.isnan(a) | (a < -1e3)
                valid_vals = a[~nodata_mask]
                if valid_vals.size > 0:
                    # normalize to 0-255
                    lo_pct, hi_pct = env_clip_pct if isinstance(env_clip_pct, (list, tuple)) else (0.0, 100.0)
                    amin = float(np.nanpercentile(valid_vals, max(0.0, lo_pct)))
                    amax = float(np.nanpercentile(valid_vals, min(100.0, hi_pct)))
                    if amax <= amin:
                        amax = amin + 1.0
                    norm = (a - amin) / (amax - amin)
                    norm = np.clip(norm, 0.0, 1.0)
                    img8 = (np.nan_to_num(norm) * 255.0).astype(np.uint8)
                    h, w = img8.shape
                    # build RGBA
                    rgb = np.dstack([img8, img8, img8])
                    alpha = (~nodata_mask).astype(np.uint8) * 255
                    rgba = np.dstack([rgb, alpha])
                    qimg = QImage(rgba.data.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
                    # Compute bbox from x/y coords
                    env_xmin = float(np.min(env_x_coords))
                    env_xmax = float(np.max(env_x_coords))
                    env_ymin = float(np.min(env_y_coords))
                    env_ymax = float(np.max(env_y_coords))
                    self._env_bbox = (env_xmin, env_xmax, env_ymin, env_ymax)
                    self._bg_qimage = qimg.copy()
                    # Lock view to environment bounds
                    self.xmin, self.xmax, self.ymin, self.ymax = env_xmin, env_xmax, env_ymin, env_ymax
                    self._live_fixed_bounds = True
                    print(f"Loaded depth from arrays: {a.shape}, bbox={self._env_bbox}")
            except Exception as e:
                print(f"Failed to load depth from arrays: {e}")
                self._bg_qimage = None
        elif env_depth is not None:
            try:
                from emergent.salmon_abm import io as _io
                arr, transform, crs = _io.enviro_import(env_depth)
                a = np.array(arr, dtype=float)
                # detect nodata-like values (common sentinel -9999)
                nodata_mask = np.isnan(a) | (a < -1e3)
                valid_vals = a[~nodata_mask]
                if valid_vals.size == 0:
                    # nothing valid
                    raise RuntimeError('env depth raster contains no valid data')
                # normalize to 0-255 using percentiles (configurable) computed on valid data
                lo_pct, hi_pct = env_clip_pct if isinstance(env_clip_pct, (list, tuple)) else (0.0, 100.0)
                amin = float(np.nanpercentile(valid_vals, max(0.0, lo_pct)))
                amax = float(np.nanpercentile(valid_vals, min(100.0, hi_pct)))
                if amax <= amin:
                    amax = amin + 1.0
                norm = (a - amin) / (amax - amin)
                norm = np.clip(norm, 0.0, 1.0)
                img8 = (np.nan_to_num(norm) * 255.0).astype(np.uint8)
                h, w = img8.shape
                # Grayscale depth raster: black (shallow) to white (deep)
                rgb = np.dstack([img8, img8, img8])
                alpha = (~nodata_mask).astype(np.uint8) * 255
                rgba = np.dstack([rgb, alpha])
                qimg = QImage(rgba.data.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
                print(f"[DEPTH RASTER] Blue colormap applied: {w}×{h} pixels, valid_range=[{amin:.2f}, {amax:.2f}]")
                # Compute raster world bbox from affine transform (a,b,c,d,e,f) mapping col,row -> x,y
                try:
                    a_t, b_t, c_t, d_t, e_t, f_t = transform
                except Exception:
                    try:
                        t = transform
                        a_t, b_t, c_t, d_t, e_t, f_t = (t.a, t.b, t.c, t.d, t.e, t.f)
                    except Exception:
                        a_t, b_t, c_t, d_t, e_t, f_t = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                cols = [0.0, float(w)]
                rows = [0.0, float(h)]
                xs = []
                ys = []
                for cc in cols:
                    for rr in rows:
                        xval = a_t * cc + b_t * rr + c_t
                        yval = d_t * cc + e_t * rr + f_t
                        xs.append(xval)
                        ys.append(yval)
                env_xmin = float(min(xs))
                env_xmax = float(max(xs))
                env_ymin = float(min(ys))
                env_ymax = float(max(ys))
                self._env_bbox = (env_xmin, env_xmax, env_ymin, env_ymax)
                self._bg_qimage = qimg.copy()
                # If an environment bbox is present, lock world view to that bbox
                try:
                    self.xmin, self.xmax, self.ymin, self.ymax = env_xmin, env_xmax, env_ymin, env_ymax
                    self._live_fixed_bounds = True
                except Exception:
                    pass
                try:
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').info('Loaded env-depth %s bbox=%s clip=%s/%s', env_depth, self._env_bbox, lo_pct, hi_pct)
                except Exception:
                    pass
            except Exception:
                self._bg_qimage = None
        # encourage smoother painter rendering
        try:
            self.setAttribute(Qt.WA_OpaquePaintEvent, False)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
            self.setUpdatesEnabled(True)
        except Exception:
            pass
        
        # Mouse interaction state
        self._mouse_drag_start = None
        self._pan_x_start = 0.0
        self._pan_y_start = 0.0
        self.setMouseTracking(False)

    def mousePressEvent(self, event):
        """Start pan on left-click drag."""
        if event.button() == Qt.LeftButton:
            self._mouse_drag_start = (event.x(), event.y())
            self._pan_x_start = self._pan_x
            self._pan_y_start = self._pan_y
            self.setCursor(Qt.ClosedHandCursor)
        event.accept()
    
    def set_positions(self, positions: np.ndarray, headings: Optional[np.ndarray] = None, 
                     battery: Optional[np.ndarray] = None, alive: Optional[np.ndarray] = None):
        """
        Load new trajectory data and start animation.
        
        Args:
            positions: (T, N, 2) array of agent positions
            headings: (T, N) array of agent headings in radians (optional)
            battery: (T, N) array of battery levels 0-1 (optional)
            alive: (T, N) boolean array of alive status (optional)
        """
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must be (T, N, 2)")
        
        # Stop any existing animation
        if self.timer.isActive():
            self.timer.stop()
        self.playing = False
        
        self.positions = positions
        self.T, self.N, _ = positions.shape
        self.heading_array = headings
        self.battery_array = battery
        self.frame = 0
        
        # DON'T recompute bounds - keep original environment bounds
        # This prevents zooming issues when agents cluster or die
        # Original bounds were set from all positions in __init__
        
        # Reset view to default
        self._pad = 1.15
        self._pan_x = 0.0
        self._pan_y = 0.0
        
        # Start playback automatically
        self.playing = True
        self.timer.start()
        self.update()
        print(f"set_positions: Loaded T={self.T}, N={self.N}, starting playback", flush=True)

    def mouseMoveEvent(self, event):
        """Pan view during drag."""
        if self._mouse_drag_start is not None:
            dx = event.x() - self._mouse_drag_start[0]
            dy = event.y() - self._mouse_drag_start[1]
            # Convert pixel movement to world fraction
            w = self.width()
            h = self.height()
            if w > 0 and h > 0:
                self._pan_x = self._pan_x_start + dx / w  # + instead of - to fix left/right reversal
                self._pan_y = self._pan_y_start - dy / h
                self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        """End pan drag."""
        if event.button() == Qt.LeftButton:
            self._mouse_drag_start = None
            self.setCursor(Qt.ArrowCursor)
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_R:
            # Reset view to default
            self._pad = 1.15
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._mouse_drag_start = None
            self._pan_x_start = 0.0
            self._pan_y_start = 0.0
            self.setCursor(Qt.ArrowCursor)
            self.update()
            print("View reset to default (R key)", flush=True)
        event.accept()

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        # Get wheel delta (positive = zoom in, negative = zoom out)
        delta = event.angleDelta().y()
        if delta != 0:
            # Adjust zoom factor (pad) - smaller pad = more zoom in
            zoom_factor = 0.9 if delta > 0 else 1.1  # Inverted: scroll up = zoom in = smaller pad
            self._pad = max(0.01, min(20.0, self._pad * zoom_factor))  # Allow 10x closer zoom
            # Clear ALL pan state to prevent corruption
            self._mouse_drag_start = None
            self._pan_x_start = self._pan_x
            self._pan_y_start = self._pan_y
            self.setCursor(Qt.ArrowCursor)
            self.update()
        event.accept()

    def _draw_fish_body(self, painter, cx, cy, heading_deg, base_radius, scale):
        """Draw a fish body as a head circle with a line extending backward.
        
        Args:
            cx, cy: screen coordinates of head center
            heading_deg: direction fish is facing in degrees
            base_radius: radius of head circle in screen pixels
            scale: screen pixels per world unit (for scaling the body line)
        """
        # Draw head circle (using currently set brush and pen)
        painter.drawEllipse(QRectF(cx - base_radius, cy - base_radius, 2 * base_radius, 2 * base_radius))
        
        # Draw body as a line extending backward from the head
        # Line length is 0.83 meters (full fish body), head radius reduced to 0.5x for better proportions
        line_length_world = 0.83  # meters (full body length)
        line_length_screen = line_length_world * scale
        
        # Convert heading to radians (heading is direction of movement)
        heading_rad = np.radians(heading_deg)
        
        # End point of line (backward from heading direction)
        # Note: Screen Y-axis is inverted (positive Y goes down), so we negate the sin component
        tail_x = cx - line_length_screen * np.cos(heading_rad)
        tail_y = cy + line_length_screen * np.sin(heading_rad)  # + instead of - for inverted Y
        
        # Draw the line (pen already set to match fish color)
        current_pen = painter.pen()
        current_pen.setWidthF(max(1.5, base_radius * 0.3))  # Line thickness
        painter.setPen(current_pen)
        painter.drawLine(QPointF(cx, cy), QPointF(tail_x, tail_y))

    def _get_battery_color(self, agent_idx: int) -> QColor:
        """Get color for agent based on battery level.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            QColor: Green (100% battery) to Red (0% battery)
        """
        if self.battery_array is None or self.frame >= len(self.battery_array) or agent_idx >= self.N:
            # Default to GREEN when no battery data (assume full charge)
            return QColor(30, 220, 30)
        
        try:
            battery = float(self.battery_array[self.frame, agent_idx])
            battery = np.clip(battery, 0.0, 1.0)
            
            # Interpolate from red (0%) to green (100%)
            # Red: (220, 30, 30)
            # Green: (30, 220, 30)
            r = int(220 - 190 * battery)
            g = int(30 + 190 * battery)
            b = 30
            
            return QColor(r, g, b)
        except Exception:
            return QColor(30, 220, 30)  # Default to green on error

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
        print(f"Start called: frame={self.frame}, T={self.T}")
        if self.frame >= self.T - 1:
            self.frame = 0
        self.playing = True
        if not self.timer.isActive():
            self.timer.start()
            print(f"Timer started with interval {self.timer.interval()}ms")

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
        import logging as _lg
        log = _lg.getLogger('realtime_viewer')
        try:
            log.debug('ReplayWidget.paintGL called: size=%sx%s frame=%s N=%s', self.width(), self.height(), getattr(self, 'frame', None), getattr(self, 'N', None))
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            try:
                painter.setRenderHint(QPainter.HighQualityAntialiasing)
            except Exception:
                pass
            w = self.width()
            h = self.height()
            
            # Fill background (dark grey for better contrast)
            painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

            # draw light grid to show canvas area
            pen = QPen(QColor(230, 230, 230))
            painter.setPen(pen)
            step = max(20, int(min(w, h) / 10))
            for x in range(0, w, step):
                painter.drawLine(x, 0, x, h)
            for y in range(0, h, step):
                painter.drawLine(0, y, w, y)

            dx = self.xmax - self.xmin
            dy = self.ymax - self.ymin
            if dx == 0:
                dx = 1.0
            if dy == 0:
                dy = 1.0

            pad_factor = float(getattr(self, '_pad', 1.15))
            
            # Maintain 1:1 aspect ratio (1 meter X = 1 meter Y on screen)
            # Start with larger data extent
            max_extent = max(dx, dy) * pad_factor
            cx = self.xmin + dx * 0.5
            cy = self.ymin + dy * 0.5
            
            # Adjust world extents to match window aspect ratio
            # This ensures 1 meter takes the same pixel count in both directions
            window_aspect = w / h if h > 0 else 1.0
            if window_aspect > 1.0:
                # Window is wider than tall - expand X extent
                dx = max_extent * window_aspect
                dy = max_extent
            else:
                # Window is taller than wide - expand Y extent
                dx = max_extent
                dy = max_extent / window_aspect
            
            xmin_loc = cx - dx * 0.5
            ymin_loc = cy - dy * 0.5
            xmax_loc = cx + dx * 0.5
            ymax_loc = cy + dy * 0.5

            sx = w / dx
            sy = h / dy
            s = min(sx, sy) * 0.95
            tx = (w - s * dx) / 2.0
            ty = (h - s * dy) / 2.0

            # pan offsets used for world->canvas mapping
            pan_x = getattr(self, '_pan_x', 0.0)
            pan_y = getattr(self, '_pan_y', 0.0)

            # draw background image mapped to the computed world rectangle using raster bbox
            if getattr(self, '_bg_qimage', None) is not None and getattr(self, '_env_bbox', None) is not None:
                try:
                    env_xmin, env_xmax, env_ymin, env_ymax = self._env_bbox
                    # compute mapping from env/world coords to canvas coords
                    def world_to_canvas(wx, wy):
                        x_adj = wx + pan_x * (xmax_loc - xmin_loc)
                        y_adj = wy + pan_y * (ymax_loc - ymin_loc)
                        sxp = tx + (x_adj - xmin_loc) * s
                        syp = ty + (ymax_loc - y_adj) * s
                        return sxp, syp

                    # get canvas rect for the env bbox corners
                    x0, y0 = world_to_canvas(env_xmin, env_ymin)
                    x1, y1 = world_to_canvas(env_xmax, env_ymax)
                    left = min(x0, x1)
                    top = min(y0, y1)
                    right = max(x0, x1)
                    bottom = max(y0, y1)
                    dest = QRectF(left, top, right - left, bottom - top)
                    # draw the background image; ensure correct orientation (flip vertically if raster rows go top->bottom)
                    try:
                        # if the original transform had negative y-scale we mirrored the image on load
                        painter.drawImage(dest, self._bg_qimage)
                    except Exception:
                        painter.drawImage(dest, self._bg_qimage)
                except Exception:
                    pass
            
            # Velocity field arrows disabled (too slow for real-time rendering)
            # TODO: Re-enable with GPU acceleration or coarser sampling
            if False and hasattr(self, '_vel_x_data') and hasattr(self, '_vel_y_data') and self._vel_x_data is not None:
                try:
                    vel_x = self._vel_x_data
                    vel_y = self._vel_y_data
                    vel_bbox = getattr(self, '_vel_bbox', None)
                    vel_transform = getattr(self, '_vel_transform', None)
                    
                    if vel_bbox is not None and vel_transform is not None:
                        # Adaptive sampling: fewer arrows when zoomed out, more when zoomed in
                        # Compute pixel size of one raster cell
                        try:
                            a_t, b_t, c_t, d_t, e_t, f_t = vel_transform
                        except:
                            t = vel_transform
                            a_t, b_t, c_t, d_t, e_t, f_t = (t.a, t.b, t.c, t.d, t.e, t.f)
                        
                        cell_width_m = abs(a_t)  # Meters per pixel
                        cell_height_m = abs(e_t)
                        
                        # Compute how many pixels one raster cell takes on screen
                        pixels_per_cell = cell_width_m * s
                        
                        # Sample every N cells based on zoom (target ~30-50 pixel spacing)
                        target_spacing_px = 40
                        sample_step = max(1, int(target_spacing_px / max(1, pixels_per_cell)))
                        
                        h_vel, w_vel = vel_x.shape
                        
                        # Helper function to convert world coords to canvas
                        def world_to_canvas_vel(wx, wy):
                            x_adj = wx + pan_x * (xmax_loc - xmin_loc)
                            y_adj = wy + pan_y * (ymax_loc - ymin_loc)
                            sxp = tx + (x_adj - xmin_loc) * s
                            syp = ty + (ymax_loc - y_adj) * s
                            return sxp, syp
                        
                        # Draw white arrows
                        arrow_pen = QPen(QColor(255, 255, 255, 180))  # Semi-transparent white
                        arrow_pen.setWidthF(1.0)
                        painter.setPen(arrow_pen)
                        
                        for row in range(0, h_vel, sample_step):
                            for col in range(0, w_vel, sample_step):
                                vx = vel_x[row, col]
                                vy = vel_y[row, col]
                                
                                # Skip nodata/invalid (nodata = -9999)
                                if not (np.isfinite(vx) and np.isfinite(vy)) or abs(vx) > 9000 or abs(vy) > 9000 or (abs(vx) + abs(vy) < 0.01):
                                    continue
                                
                                # Compute world coords of this raster cell center
                                wx = a_t * (col + 0.5) + b_t * (row + 0.5) + c_t
                                wy = d_t * (col + 0.5) + e_t * (row + 0.5) + f_t
                                
                                # Convert to canvas coords
                                cx, cy = world_to_canvas_vel(wx, wy)
                                
                                # Arrow length in pixels (scale by velocity magnitude)
                                vel_mag = np.sqrt(vx**2 + vy**2)
                                arrow_length = min(30, vel_mag * s * 5)  # Scale arrows
                                
                                # Arrow endpoint
                                arrow_dx = (vx / vel_mag) * arrow_length if vel_mag > 0 else 0
                                arrow_dy = -(vy / vel_mag) * arrow_length if vel_mag > 0 else 0  # Negative because canvas Y is inverted
                                
                                ex = cx + arrow_dx
                                ey = cy + arrow_dy
                                
                                # Draw arrow line
                                painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))
                                
                                # Draw arrowhead (small triangle)
                                if arrow_length > 5:
                                    angle = np.arctan2(arrow_dy, arrow_dx)
                                    head_size = 4
                                    angle1 = angle + 2.8  # ~160 degrees
                                    angle2 = angle - 2.8
                                    
                                    h1x = ex + head_size * np.cos(angle1)
                                    h1y = ey + head_size * np.sin(angle1)
                                    h2x = ex + head_size * np.cos(angle2)
                                    h2y = ey + head_size * np.sin(angle2)
                                    
                                    painter.drawLine(QPointF(ex, ey), QPointF(h1x, h1y))
                                    painter.drawLine(QPointF(ex, ey), QPointF(h2x, h2y))
                except Exception as e:
                    pass  # Silently skip if velocity rendering fails

            pts = self.positions[self.frame]
            if self.N == 0 or pts.size == 0:
                painter.setPen(QPen(QColor(200, 200, 200)))
                painter.drawText(int(w / 2) - 80, int(h / 2), 'Waiting for frames...')
                painter.setPen(QPen(QColor(180, 180, 180)))
                painter.drawText(6, 28, f'DEBUG: frame={self.frame} N={self.N}')
                painter.end()
                return

            pen = QPen(QColor(180, 10, 10))
            pen.setWidthF(1.0)
            painter.setPen(pen)
            brush_col = QColor(220, 30, 30)
            painter.setBrush(brush_col)
            # default radius scales with canvas; override if user provided `point_size`
            # Reduced head size for better body/head proportions (was 0.002, now 0.001)
            r = max(2, int(min(w, h) * 0.001))  # Head size: 0.1% of screen
            if self._point_size is not None:
                r = max(2, int(self._point_size * 0.5))  # Halve user-provided size for smaller heads
            if getattr(self, '_debug_force_big', False):
                r = max(r, int(min(w, h) * 0.01))

            pan_x = getattr(self, '_pan_x', 0.0)
            pan_y = getattr(self, '_pan_y', 0.0)
            # draw trail buffer first (older frames with fainter alpha)
            try:
                if hasattr(self, '_trail_buf') and len(self._trail_buf) > 0:
                    # draw older entries first
                    for ti, tpts in enumerate(self._trail_buf[:-1]):
                        alpha = int(80 * (ti + 1) / max(1, len(self._trail_buf)))
                        trail_brush = QColor(220, 30, 30)
                        trail_brush.setAlpha(alpha)
                        painter.setBrush(trail_brush)
                        for j in range(tpts.shape[0]):
                            x, y = tpts[j]
                            if not (np.isfinite(x) and np.isfinite(y)):
                                continue
                            x_adj = x + pan_x * (xmax_loc - xmin_loc)
                            y_adj = y + pan_y * (ymax_loc - ymin_loc)
                            sxp = tx + (x_adj - xmin_loc) * s
                            syp = ty + (ymax_loc - y_adj) * s
                            painter.drawEllipse(QRectF(sxp - r, syp - r, 2 * r, 2 * r))
                    # restore main brush for newest points
                    painter.setBrush(brush_col)
            except Exception:
                pass

            # draw current points (newest) as fish bodies
            n_to_draw = self.N if self.max_agents is None else min(self.N, self.max_agents)
            
            # DEBUG: Log when max_agents changes (sparse sampling to avoid spam)
            if not hasattr(self, '_last_logged_max_agents'):
                self._last_logged_max_agents = None
            if self._last_logged_max_agents != self.max_agents:
                print(f"paintGL: N={self.N}, max_agents={self.max_agents}, n_to_draw={n_to_draw}", flush=True)
                self._last_logged_max_agents = self.max_agents
            
            for i in range(n_to_draw):
                x, y = pts[i]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                x_adj = x + pan_x * (xmax_loc - xmin_loc)
                y_adj = y + pan_y * (ymax_loc - ymin_loc)
                sxp = tx + (x_adj - xmin_loc) * s
                syp = ty + (ymax_loc - y_adj) * s
                
                # Check if fish is dead (if alive_array available)
                is_dead = False
                if hasattr(self, 'alive_array') and self.alive_array is not None:
                    if self.frame < len(self.alive_array) and i < self.alive_array.shape[1]:
                        is_dead = not self.alive_array[self.frame, i]
                
                # Dead fish are white, alive fish colored by battery level
                if is_dead:
                    fish_color = QColor(255, 255, 255)  # White for dead
                    pen = QPen(QColor(200, 200, 200))  # Light grey outline
                else:
                    fish_color = self._get_battery_color(i)  # Battery gradient
                    pen = QPen(fish_color.darker(120))
                
                painter.setBrush(fish_color)
                pen.setWidthF(1.0)
                painter.setPen(pen)
                
                # Get heading from heading_array if available, otherwise compute from velocity
                heading_deg = 0.0
                heading_computed = False
                
                # Always compute from velocity when available (more accurate than stored heading)
                if self.frame > 0 and self.frame < len(self.positions):
                    prev_pts = self.positions[self.frame - 1]
                    if i < prev_pts.shape[0]:
                        prev_x, prev_y = prev_pts[i]
                        if np.isfinite(prev_x) and np.isfinite(prev_y):
                            dx = x - prev_x
                            dy = y - prev_y
                            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                                heading_deg = np.degrees(np.arctan2(dy, dx))
                                heading_computed = True
                
                # Fallback to stored heading only if velocity-based computation failed
                if not heading_computed and self.heading_array is not None and self.frame < len(self.heading_array) and i < self.heading_array.shape[1]:
                    heading_rad = self.heading_array[self.frame, i]
                    if np.isfinite(heading_rad) and abs(heading_rad) > 1e-6:
                        heading_deg = np.degrees(heading_rad)
                
                # Draw fish body with line
                self._draw_fish_body(painter, sxp, syp, heading_deg, r, s)

            painter.setPen(QPen(QColor(0, 0, 0)))
            # display a live frame counter when receiving live updates (T often == 1)
            try:
                if getattr(self, '_live_count', 0) > 0 and self.T == 1:
                    # draw a prominent live badge in top-right
                    try:
                        badge_text = f"LIVE {self._live_count}  Agents: {self.N}"
                        fm = painter.fontMetrics()
                        bw = fm.width(badge_text) + 12
                        bh = fm.height() + 6
                        rx = w - bw - 8
                        ry = 8
                        painter.setBrush(QColor(200, 30, 30, 200))
                        painter.setPen(QPen(QColor(180, 20, 20)))
                        painter.drawRoundedRect(rx, ry, bw, bh, 6, 6)
                        painter.setPen(QPen(QColor(255, 255, 255)))
                        painter.drawText(rx + 6, ry + bh - 6, badge_text)
                    except Exception:
                        painter.drawText(6, 14, f"Live frames: {self._live_count}  Agents: {self.N}")
                else:
                    painter.drawText(6, 14, f"Frame: {self.frame+1}/{self.T}  Agents: {self.N}")
            except Exception:
                painter.drawText(6, 14, f"Frame: {self.frame+1}/{self.T}  Agents: {self.N}")
            
            # Display projected coordinate bounds (real world coordinates)
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(6, h - 6, f'X: {xmin_loc:.1f}m')
            painter.drawText(w - 120, h - 6, f'X: {xmax_loc:.1f}m')
            painter.drawText(6, 42, f'Y: {ymax_loc:.1f}m')
            painter.drawText(6, h - 20, f'Y: {ymin_loc:.1f}m')
            
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(6, 28, f'DEBUG: frame={self.frame} N={self.N} xmin={xmin_loc:.2f} xmax={xmax_loc:.2f} ymin={ymin_loc:.2f} ymax={ymax_loc:.2f}')
            painter.end()
        except Exception:
            _lg.getLogger('realtime_viewer').exception('ReplayWidget.paintGL failed')

    def update_frame(self, arr):
        """Called from GUI thread with an (N,2) float array to update current points (live mode)."""
        try:
            pts = np.asarray(arr, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                return
            # update world bounds: initialize on first live frame, then expand only
            xs = pts[:, 0]
            ys = pts[:, 1]
            valid = np.isfinite(xs) & np.isfinite(ys)
            if np.any(valid):
                # If an environment bbox is present, do not change view bounds (lock to env)
                if getattr(self, '_env_bbox', None) is not None:
                    # do not modify xmin/xmax/ymin/ymax
                    pass
                else:
                    new_xmin = float(np.nanmin(xs[valid]))
                    new_xmax = float(np.nanmax(xs[valid]))
                    new_ymin = float(np.nanmin(ys[valid]))
                    new_ymax = float(np.nanmax(ys[valid]))
                    if getattr(self, '_live_fixed_bounds', False):
                        # by default do not expand bounds to avoid zooming out; expand only if allowed
                        if self._allow_expand_bounds:
                            self.xmin = min(self.xmin, new_xmin)
                            self.xmax = max(self.xmax, new_xmax)
                            self.ymin = min(self.ymin, new_ymin)
                            self.ymax = max(self.ymax, new_ymax)
                    else:
                        self.xmin = new_xmin
                        self.xmax = new_xmax
                        self.ymin = new_ymin
                        self.ymax = new_ymax
                        self._live_fixed_bounds = True
            # assign into positions buffer so paintGL can access indexed by frame
            # apply optional temporal smoothing to reduce jitter
            try:
                if self._smooth_alpha is not None and 0.0 <= self._smooth_alpha < 1.0:
                    if self._smoothed_positions is None:
                        self._smoothed_positions = pts.copy()
                    else:
                        self._smoothed_positions = (self._smooth_alpha * pts) + ((1.0 - self._smooth_alpha) * self._smoothed_positions)
                    use_pts = self._smoothed_positions
                else:
                    use_pts = pts
                self.positions = use_pts[np.newaxis, :, :]
                self.T, self.N = self.positions.shape[0], self.positions.shape[1]
                # update trail buffer
                try:
                    if not hasattr(self, '_trail_buf'):
                        self._trail_buf = []
                    # append the raw incoming points (not smoothed) to preserve motion trail fidelity
                    self._trail_buf.append(pts.copy())
                    if len(self._trail_buf) > getattr(self, '_trail_length', 8):
                        self._trail_buf.pop(0)
                except Exception:
                    pass
            except Exception:
                pass
            # request repaint
            self.update()
        except Exception:
            pass


class GLViewer(QOpenGLWidget):
    """OpenGL VBO-backed viewer for large numbers of agents.

    This attempts to use PyOpenGL. If not available, constructing this
    widget will raise ImportError and the caller should fall back.
    """

    def __init__(self, positions: np.ndarray, parent: Optional[QWidget] = None, pad: float = 1.15, force_vbo: bool = False, point_size: Optional[float] = None, env_depth_array: Optional[np.ndarray] = None, env_x_coords: Optional[np.ndarray] = None, env_y_coords: Optional[np.ndarray] = None, battery_array: Optional[np.ndarray] = None, heading_array: Optional[np.ndarray] = None):
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
        self._pad = float(pad)
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._mouse_drag_start = None
        self._pan_x_start = 0.0
        self._pan_y_start = 0.0
        self._force_vbo = bool(force_vbo)
        self._point_size = None if point_size is None else float(point_size)
        self.env_depth_array = env_depth_array
        self.env_x_coords = env_x_coords
        self.env_y_coords = env_y_coords
        self._env_texture = None
        self.battery_array = battery_array  # Battery data for fatigue visualization
        self.heading_array = heading_array  # Heading data for oriented fish rendering
        self.max_agents = None  # Maximum number of agents to display (None = show all)

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

    def mousePressEvent(self, event):
        """Start pan on left-click drag."""
        try:
            from PyQt5.QtCore import Qt
            if event.button() == Qt.LeftButton:
                self._mouse_drag_start = (event.x(), event.y())
                self._pan_x_start = self._pan_x
                self._pan_y_start = self._pan_y
                self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        except Exception:
            pass

    def mouseMoveEvent(self, event):
        """Pan view during drag."""
        try:
            if hasattr(self, '_mouse_drag_start') and self._mouse_drag_start is not None:
                dx = event.x() - self._mouse_drag_start[0]
                dy = event.y() - self._mouse_drag_start[1]
                w = self.width()
                h = self.height()
                if w > 0 and h > 0:
                    self._pan_x = self._pan_x_start - dx / w  # Negate dx: drag right = pan right
                    self._pan_y = self._pan_y_start + dy / h  # Positive dy = drag down = pan down
                    self.update()
            event.accept()
        except Exception:
            pass

    def mouseReleaseEvent(self, event):
        """End pan on left-click release."""
        try:
            from PyQt5.QtCore import Qt
            if event.button() == Qt.LeftButton:
                self._mouse_drag_start = None
                self.setCursor(Qt.ArrowCursor)
            event.accept()
        except Exception:
            pass

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        try:
            # Clear any active drag to prevent scroll from affecting pan
            self._mouse_drag_start = None
            delta = event.angleDelta().y()
            if delta != 0:
                zoom_factor = 0.9 if delta > 0 else 1.1
                self._pad = max(0.01, min(20.0, self._pad * zoom_factor))  # Allow 10x closer zoom
                self.update()
            event.accept()
        except Exception:
            pass

    def initializeGL(self):
        if not self._gl_available:
            return
        GL = self._GL
        try:
            GL.glClearColor(0.12, 0.12, 0.12, 1.0)  # Dark grey background
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)  # Allow shader to control point size
            import ctypes
            # Preallocate a GPU buffer (raw GL buffer) for dynamic point data
            self._vbo_capacity = max(256, int(getattr(self, 'N', 0)))
            self._vbo_capacity_bytes = int(self._vbo_capacity * 2 * np.dtype(np.float32).itemsize)
            
            # Also preallocate color buffer for battery visualization
            self._color_vbo_capacity_bytes = int(self._vbo_capacity * 3 * np.dtype(np.float32).itemsize)

            # attempt raw GL buffer creation
            self._vbo_id = None
            self._color_vbo_id = None
            try:
                self._vbo_id = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._vbo_id))
                GL.glBufferData(GL.GL_ARRAY_BUFFER, self._vbo_capacity_bytes, None, GL.GL_DYNAMIC_DRAW)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
                # Create color VBO
                self._color_vbo_id = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._color_vbo_id))
                GL.glBufferData(GL.GL_ARRAY_BUFFER, self._color_vbo_capacity_bytes, None, GL.GL_DYNAMIC_DRAW)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                
                self._using_raw_vbo = True
            except Exception:
                # try PyOpenGL VBO wrapper
                try:
                    self._vbo = self._glvbo.VBO(np.zeros((self._vbo_capacity, 2), dtype=np.float32))
                    self._using_raw_vbo = False
                except Exception:
                    self._vbo = None
                    self._using_raw_vbo = False

            # compile a minimal passthrough shader (optional)
            vs = b"""
            #version 120
            attribute vec2 position;
            attribute vec3 color;
            varying vec3 vColor;
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * vec4(position.xy, 0.0, 1.0);
                gl_PointSize = 5.0;
                vColor = color;
            }
            """
            fs = b"""
            #version 120
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
            """
            try:
                self._program = GL.glCreateProgram()
                vs_id = GL.glCreateShader(GL.GL_VERTEX_SHADER)
                fs_id = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
                GL.glShaderSource(vs_id, vs)
                GL.glCompileShader(vs_id)
                GL.glShaderSource(fs_id, fs)
                GL.glCompileShader(fs_id)
                GL.glAttachShader(self._program, vs_id)
                GL.glAttachShader(self._program, fs_id)
                GL.glLinkProgram(self._program)
            except Exception:
                self._program = None
            
            # Create texture from environment depth array if available
            if self.env_depth_array is not None and self.env_x_coords is not None and self.env_y_coords is not None:
                try:
                    # Normalize depth to 0-1 range for visualization
                    depth_array = np.array(self.env_depth_array, dtype=float)
                    depth_min = np.nanmin(depth_array)
                    depth_max = np.nanmax(depth_array)
                    if depth_max > depth_min:
                        depth_norm = (depth_array - depth_min) / (depth_max - depth_min)
                    else:
                        depth_norm = np.zeros_like(depth_array)
                    
                    # Create RGB image (blue gradient for depth)
                    # Deeper water = darker blue, shallower = lighter blue/white
                    h, w = depth_norm.shape
                    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                    rgb_image[:, :, 0] = (200 * depth_norm).astype(np.uint8)  # R
                    rgb_image[:, :, 1] = (220 * depth_norm).astype(np.uint8)  # G
                    rgb_image[:, :, 2] = (255 * depth_norm).astype(np.uint8)  # B
                    
                    # Flip vertically for OpenGL texture coordinates (OpenGL origin bottom-left)
                    rgb_image = np.flipud(rgb_image)
                    
                    # Store environment bounds
                    self.env_xmin = float(np.min(self.env_x_coords))
                    self.env_xmax = float(np.max(self.env_x_coords))
                    self.env_ymin = float(np.min(self.env_y_coords))
                    self.env_ymax = float(np.max(self.env_y_coords))
                    
                    # Create OpenGL texture
                    self._env_texture = GL.glGenTextures(1)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._env_texture)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb_image)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').info(f'Created environment texture: {w}x{h}, ID={self._env_texture}, bounds=({self.env_xmin:.1f},{self.env_ymin:.1f})->({self.env_xmax:.1f},{self.env_ymax:.1f})')
                except Exception as tex_ex:
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').exception('Failed to create environment texture')
                    self._env_texture = None
        except Exception:
            import logging as _lg
            _lg.getLogger('realtime_viewer').exception('GLViewer.initializeGL failed')

    def paintGL(self):
        if not self._gl_available:
            # fallback to software painter
            painter = QPainter(self)
            painter.drawText(10, 20, 'OpenGL not available')
            painter.end()
            return
        GL = self._GL
        # ensure viewport and clear
        GL.glViewport(0, 0, int(self.width()), int(self.height()))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # fetch current positions and draw as points
        try:
            pts = self.positions[self.frame].astype(np.float32)
            # set projection to world coordinates
            try:
                GL.glMatrixMode(GL.GL_PROJECTION)
                GL.glLoadIdentity()
                # Note: OpenGL bottom-left origin; flip Y by swapping ymin/ymax for correct orientation
                # add padding to world bounds so view is slightly zoomed out
                try:
                    dx = self.xmax - self.xmin
                    dy = self.ymax - self.ymin
                    if dx == 0:
                        dx = 1.0
                    if dy == 0:
                        dy = 1.0
                    pad = float(getattr(self, '_pad', 1.15))
                    pan_x = float(getattr(self, '_pan_x', 0.0))
                    pan_y = float(getattr(self, '_pan_y', 0.0))
                    
                    # Maintain 1:1 aspect ratio (1 meter X = 1 meter Y on screen)
                    max_extent = max(dx, dy) * pad
                    cx = self.xmin + dx * 0.5
                    cy = self.ymin + dy * 0.5
                    
                    # Adjust world extents to match window aspect ratio
                    w_viewport = float(self.width())
                    h_viewport = float(self.height())
                    window_aspect = w_viewport / h_viewport if h_viewport > 0 else 1.0
                    
                    if window_aspect > 1.0:
                        # Window is wider than tall - expand X extent
                        world_dx = max_extent * window_aspect
                        world_dy = max_extent
                    else:
                        # Window is taller than wide - expand Y extent
                        world_dx = max_extent
                        world_dy = max_extent / window_aspect
                    
                    # Apply pan offsets as fraction of world extents
                    cx += pan_x * world_dx
                    cy += pan_y * world_dy
                    
                    xminp = cx - world_dx * 0.5
                    xmaxp = cx + world_dx * 0.5
                    yminp = cy - world_dy * 0.5
                    ymaxp = cy + world_dy * 0.5
                except Exception:
                    xminp = self.xmin
                    xmaxp = self.xmax
                    yminp = self.ymin
                    ymaxp = self.ymax
                GL.glOrtho(xminp, xmaxp, yminp, ymaxp, -1.0, 1.0)
                GL.glMatrixMode(GL.GL_MODELVIEW)
                GL.glLoadIdentity()
            except Exception:
                pass
            
            # Render environment background texture if available
            if self._env_texture is not None:
                try:
                    # Make sure no shader is active for texture rendering
                    GL.glUseProgram(0)
                    GL.glDisable(GL.GL_BLEND)
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glEnable(GL.GL_TEXTURE_2D)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self._env_texture)
                    GL.glColor3f(1.0, 1.0, 1.0)
                    
                    # Draw textured quad covering environment bounds
                    env_xmin = getattr(self, 'env_xmin', self.xmin)
                    env_xmax = getattr(self, 'env_xmax', self.xmax)
                    env_ymin = getattr(self, 'env_ymin', self.ymin)
                    env_ymax = getattr(self, 'env_ymax', self.ymax)
                    
                    import logging as _lg
                    log = _lg.getLogger('realtime_viewer')
                    log.debug(f'Rendering texture quad: bounds=({env_xmin:.1f},{env_ymin:.1f})->({env_xmax:.1f},{env_ymax:.1f})')
                    
                    GL.glBegin(GL.GL_QUADS)
                    GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(env_xmin, env_ymin)
                    GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(env_xmax, env_ymin)
                    GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(env_xmax, env_ymax)
                    GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(env_xmin, env_ymax)
                    GL.glEnd()
                    
                    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    GL.glDisable(GL.GL_TEXTURE_2D)
                except Exception:
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').exception('Failed to render environment texture')
            
            # choose drawing path: prefer raw GL buffer when available/forced, else wrapper or immediate
            import logging as _lg
            log = _lg.getLogger('realtime_viewer')
            try:
                pts2 = np.ascontiguousarray(pts, dtype=np.float32)
                npoints = pts2.shape[0]
                using_raw = getattr(self, '_using_raw_vbo', False) and getattr(self, '_vbo_id', None) is not None
                using_wrapper = getattr(self, '_vbo', None) is not None
                if self._force_vbo and using_raw:
                    path = 'raw-vbo'
                elif using_raw and not using_wrapper:
                    path = 'raw-vbo'
                elif using_wrapper and not using_raw:
                    path = 'wrapper-vbo'
                elif using_raw and using_wrapper:
                    path = 'raw-vbo'
                else:
                    path = 'immediate'
                log.debug('GLViewer.paintGL draw path=%s pts=%s bounds=(%s,%s,%s,%s)', path, getattr(pts, 'shape', None), self.xmin, self.xmax, self.ymin, self.ymax)

                if path == 'raw-vbo':
                    import ctypes
                    
                    # Build line segments for fish bodies (head point + tail point per fish)
                    line_vertices = np.zeros((npoints * 2, 2), dtype=np.float32)  # 2 vertices per fish
                    line_colors = np.zeros((npoints * 2, 3), dtype=np.float32)    # Color for each vertex
                    
                    line_length = 0.415  # meters (half of 0.83m)
                    
                    for i in range(npoints):
                        x, y = pts2[i]
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        
                        # Get heading from heading_array if available
                        heading_rad = 0.0
                        if self.heading_array is not None and self.frame < self.heading_array.shape[0] and i < self.heading_array.shape[1]:
                            heading_rad = float(self.heading_array[self.frame, i])
                            if not np.isfinite(heading_rad):
                                heading_rad = 0.0
                        
                        # Get battery color
                        if self.battery_array is not None and self.frame < self.battery_array.shape[0] and i < self.battery_array.shape[1]:
                            battery = float(self.battery_array[self.frame, i])
                            battery = np.clip(battery, 0.0, 1.0)
                            r = (220 - 190 * battery) / 255.0
                            g = (30 + 190 * battery) / 255.0
                            b = 30 / 255.0
                        else:
                            r, g, b = 0.8, 0.12, 0.12
                        
                        # Head position (current position)
                        line_vertices[i*2] = [x, y]
                        line_colors[i*2] = [r, g, b]
                        
                        # Tail position (extending backward from heading)
                        tail_x = x - line_length * np.cos(heading_rad)
                        tail_y = y - line_length * np.sin(heading_rad)
                        line_vertices[i*2 + 1] = [tail_x, tail_y]
                        line_colors[i*2 + 1] = [r, g, b]
                    
                    line_vertices = np.ascontiguousarray(line_vertices, dtype=np.float32)
                    line_colors = np.ascontiguousarray(line_colors, dtype=np.float32)
                    
                    # Upload vertex data (line segments)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._vbo_id))
                    size_bytes = line_vertices.nbytes
                    if size_bytes <= getattr(self, '_vbo_capacity_bytes', 0):
                        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, line_vertices)
                    else:
                        self._vbo_capacity = npoints * 2  # 2 vertices per fish
                        self._vbo_capacity_bytes = line_vertices.nbytes
                        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._vbo_capacity_bytes, line_vertices, GL.GL_DYNAMIC_DRAW)
                    
                    # Upload color data (per vertex)
                    if self._color_vbo_id is not None:
                        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._color_vbo_id))
                        color_size_bytes = line_colors.nbytes
                        if color_size_bytes <= getattr(self, '_color_vbo_capacity_bytes', 0):
                            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, line_colors)
                        else:
                            self._color_vbo_capacity_bytes = line_colors.nbytes
                            GL.glBufferData(GL.GL_ARRAY_BUFFER, self._color_vbo_capacity_bytes, line_colors, GL.GL_DYNAMIC_DRAW)
                    
                    if getattr(self, '_program', None) is not None:
                        GL.glUseProgram(self._program)
                        
                        # Bind position attribute for heads (points)
                        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._vbo_id))
                        loc = GL.glGetAttribLocation(self._program, b'position')
                        if loc != -1:
                            GL.glEnableVertexAttribArray(loc)
                            GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
                        
                        # Bind color attribute for heads
                        if self._color_vbo_id is not None:
                            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._color_vbo_id))
                            color_loc = GL.glGetAttribLocation(self._program, b'color')
                            if color_loc != -1:
                                GL.glEnableVertexAttribArray(color_loc)
                                GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
                        
                        # Draw head points (every other vertex is a head)
                        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
                        # Extract just the head vertices (every even index: 0, 2, 4, ...)
                        for i in range(npoints):
                            GL.glDrawArrays(GL.GL_POINTS, i*2, 1)  # Draw one point per fish head
                        
                        # Draw body lines
                        GL.glLineWidth(2.0)
                        GL.glDrawArrays(GL.GL_LINES, 0, npoints * 2)  # Draw all line segments
                        
                        if loc != -1:
                            GL.glDisableVertexAttribArray(loc)
                        if self._color_vbo_id is not None and color_loc != -1:
                            GL.glDisableVertexAttribArray(color_loc)
                        
                        GL.glUseProgram(0)
                    else:
                        # Fallback to fixed function pipeline with color array
                        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._vbo_id))
                        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                        GL.glVertexPointer(2, GL.GL_FLOAT, 0, ctypes.c_void_p(0))
                        
                        if self._color_vbo_id is not None:
                            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(self._color_vbo_id))
                            GL.glEnableClientState(GL.GL_COLOR_ARRAY)
                            GL.glColorPointer(3, GL.GL_FLOAT, 0, ctypes.c_void_p(0))
                        
                        # Draw head points
                        GL.glPointSize(5.0)
                        for i in range(npoints):
                            GL.glDrawArrays(GL.GL_POINTS, i*2, 1)  # Draw one point per fish head
                        
                        # Draw body lines
                        GL.glLineWidth(2.0)
                        GL.glDrawArrays(GL.GL_LINES, 0, npoints * 2)  # Draw all line segments
                        
                        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
                        if self._color_vbo_id is not None:
                            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
                    
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

                elif path == 'wrapper-vbo':
                    self._vbo.set_array(pts2)
                    self._vbo.bind()
                    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                    GL.glVertexPointer(2, GL.GL_FLOAT, 0, self._vbo)
                    GL.glPointSize(3.0)
                    GL.glDrawArrays(GL.GL_POINTS, 0, npoints)
                    GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
                    self._vbo.unbind()

                else:
                    # Immediate mode rendering with battery-based colors
                    GL.glPointSize(6.0)
                    GL.glBegin(GL.GL_POINTS)
                    for i, (x, y) in enumerate(pts):
                        # Get battery color for this agent
                        if self.battery_array is not None and self.frame < self.battery_array.shape[0] and i < self.battery_array.shape[1]:
                            try:
                                battery = float(self.battery_array[self.frame, i])
                                battery = np.clip(battery, 0.0, 1.0)
                                r = (220 - 190 * battery) / 255.0
                                g = (30 + 190 * battery) / 255.0
                                b = 30 / 255.0
                                GL.glColor3f(r, g, b)
                            except Exception:
                                GL.glColor3f(0.8, 0.12, 0.12)  # fallback to red
                        else:
                            GL.glColor3f(0.8, 0.12, 0.12)  # default red if no battery data
                        GL.glVertex2f(float(x), float(y))
                    GL.glEnd()
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
                import logging as _lg
                _lg.getLogger('realtime_viewer').exception('GLViewer.paintGL failed')
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
    def __init__(self, positions: np.ndarray, watchdog_seconds: float = 5.0, pad: float = 1.15, force_vbo: bool = False, point_size: Optional[float] = None):
        super().__init__()
        self.setWindowTitle("Realtime Simulation Viewer")
        self._watchdog_seconds = float(watchdog_seconds)
        self._point_size = point_size
        self.viewer = ReplayWidget(positions, point_size=point_size)
        # Set initial max_agents to all agents
        self.viewer.max_agents = positions.shape[1]
        # view state
        self._pad = float(pad)
        self._force_vbo = bool(force_vbo)
        self._pan_x = 0.0
        self._pan_y = 0.0

        btn_start = QPushButton("Start")
        btn_pause = QPushButton("Pause")
        btn_stop = QPushButton("Stop")
        btn_restart = QPushButton("Restart")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 400)
        self.speed_slider.setValue(100)
        lbl_speed = QLabel("Speed")
        
        # Agent count control
        from PyQt5.QtWidgets import QSpinBox
        lbl_agents = QLabel("Max Agents:")
        self.agent_count_spin = QSpinBox()
        self.agent_count_spin.setRange(1, 100000)
        self.agent_count_spin.setValue(positions.shape[1])  # Default to all agents
        self.agent_count_spin.setToolTip("Maximum number of agents to display")
        self.agent_count_spin.valueChanged.connect(self._on_agent_count_changed)
        
        # Timesteps control
        lbl_timesteps = QLabel("Max Timesteps:")
        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(1, 100000)
        self.timesteps_spin.setValue(positions.shape[0])  # Default to all timesteps
        self.timesteps_spin.setToolTip("Maximum number of timesteps to display")
        self.timesteps_spin.valueChanged.connect(self._on_timesteps_changed)

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
        hl.addWidget(lbl_agents)
        hl.addWidget(self.agent_count_spin)
        hl.addWidget(lbl_timesteps)
        hl.addWidget(self.timesteps_spin)
        hl.addStretch()
        
        # Mouse controls: Left-click drag to pan, scroll wheel to zoom

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
    
    def _on_agent_count_changed(self, value: int):
        """Update max agents displayed in viewer."""
        print(f"Agent count changed to: {value}", flush=True)
        if hasattr(self.viewer, 'max_agents'):
            self.viewer.max_agents = value
            print(f"Viewer max_agents set to: {self.viewer.max_agents}, N={self.viewer.N}", flush=True)
            self.viewer.repaint()  # Force immediate repaint instead of update()
        else:
            print(f"WARNING: Viewer does not have max_agents attribute", flush=True)
    
    def _on_timesteps_changed(self, value: int):
        """Update max timesteps displayed in viewer."""
        print(f"Timesteps changed to: {value}", flush=True)
        if hasattr(self.viewer, 'positions') and self.viewer.positions is not None:
            # Limit the number of timesteps by truncating the positions array
            original_T = self.viewer.positions.shape[0]
            new_T = min(value, original_T)
            if hasattr(self.viewer, 'T'):
                self.viewer.T = new_T
                print(f"Viewer T set to: {self.viewer.T}", flush=True)
                # Reset to frame 0 if current frame is beyond new limit
                if self.viewer.frame >= new_T:
                    self.viewer.frame = 0
                self.viewer.update()
        else:
            print(f"WARNING: Viewer does not have positions array", flush=True)


def swap_viewer_in_main(win: MainWindow, new_widget: QWidget):
    """Safely replace the viewer widget in the main window without collapsing layout.

    This sets sensible size policies and removes the old widget cleanly.
    """
    try:
        old = win.viewer
        parent = win.centralWidget()
        layout = parent.layout()
        # ensure new widget inherits sizing from old widget
        new_widget.setMinimumSize(old.minimumSize())
        new_widget.setSizePolicy(old.sizePolicy())
        # perform replace
        layout.replaceWidget(old, new_widget)
        try:
            old.hide()  # Hide the old viewer
        except Exception:
            pass
        try:
            old.setParent(None)  # Remove old viewer from parent
        except Exception:
            pass
        win.viewer = new_widget
        new_widget.show()
        
        # Reconnect buttons and controls to the new viewer
        try:
            # Reconnect agent count spinbox and set initial value
            from PyQt5.QtWidgets import QSpinBox
            agent_spin = None
            for child in win.centralWidget().findChildren(QSpinBox):
                # Find the agent count spinbox by checking if it has the right range
                if child.minimum() == 1 and child.maximum() == 100000:
                    agent_spin = child
                    break
            
            if agent_spin is not None:
                # Set initial max_agents from current spinbox value
                new_widget.max_agents = agent_spin.value()
                # Reconnect the signal
                try:
                    agent_spin.valueChanged.disconnect()
                except:
                    pass
                agent_spin.valueChanged.connect(lambda value: setattr(new_widget, 'max_agents', value) or new_widget.update())
            
            # Find the buttons and reconnect them
            for child in win.centralWidget().findChildren(QPushButton):
                if child.text() == "Start":
                    child.clicked.disconnect()
                    child.clicked.connect(new_widget.start)
                elif child.text() == "Pause":
                    child.clicked.disconnect()
                    child.clicked.connect(new_widget.pause)
                elif child.text() == "Stop":
                    child.clicked.disconnect()
                    child.clicked.connect(new_widget.stop)
                elif child.text() == "Restart":
                    child.clicked.disconnect()
                    child.clicked.connect(new_widget.restart)
        except Exception as e:
            print(f"Warning: Could not reconnect controls: {e}")
    except Exception:
        # best-effort fallback
        try:
            win.centralWidget().layout().replaceWidget(win.viewer, new_widget)
            try:
                win.viewer.setParent(None)
            except Exception:
                pass
            win.viewer = new_widget
        except Exception:
            pass


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
        # improve TCP behavior for low-latency streaming
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass
        # retry connect with visible logging
        connected = False
        attempts = 0
        max_attempts = 60
        while attempts < max_attempts:
            try:
                attempts += 1
                sock.connect((self.host, self.port))
                connected = True
                break
            except Exception as e:
                import logging as _lg
                _lg.getLogger('realtime_viewer').debug('LiveReceiver connect attempt %d/%d failed: %s', attempts, max_attempts, e)
                time.sleep(0.2)
        if not connected:
            try:
                self.error.emit(f'connect_failed:{self.host}:{self.port}')
            except Exception:
                pass
            return
        try:
            peer = sock.getpeername()
            import logging as _lg
            _lg.getLogger('realtime_viewer').info('LiveReceiver connected to %s:%d', peer[0], peer[1])
        except Exception:
            pass
        sock.setblocking(True)
        self._sock = sock
        try:
            while self._running:
                first = sock.recv(1)
                if not first:
                    break

                # If first byte is 'R' -> raw protocol: read next 4-byte length
                if first == b'R':
                    lb = sock.recv(4)
                    if not lb or len(lb) < 4:
                        self.error.emit('incomplete_raw_length')
                        break
                    (nbytes,) = struct.unpack('!I', lb)
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').debug('raw header nbytes=%d', nbytes)
                    buf = bytearray()
                    while len(buf) < nbytes:
                        chunk = sock.recv(nbytes - len(buf))
                        if not chunk:
                            break
                        buf.extend(chunk)
                    if len(buf) < nbytes:
                        self.error.emit('incomplete_raw_payload')
                        _lg.getLogger('realtime_viewer').debug('incomplete_raw_payload expected=%d got=%d', nbytes, len(buf))
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
                    import logging as _lg
                    _lg.getLogger('realtime_viewer').debug('npy header length=%d', nbytes)
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
                        self.error.emit(f'npy_load_error:{e}')
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
    parser.add_argument("--view-pad", dest="view_pad", type=float, default=1.15, help="View padding factor (zoom out). Default 1.15")
    parser.add_argument("--force-vbo", dest="force_vbo", action="store_true", help="Force VBO/raw GL buffer path when available")
    parser.add_argument("--point-size", dest="point_size", type=float, default=None, help="Point radius in pixels for agent rendering (overrides default scaling)")
    parser.add_argument("--env-depth", dest="env_depth", type=str, default=None, help="Path to depth GeoTIFF to render as background")
    parser.add_argument("--smooth-alpha", dest="smooth_alpha", type=float, default=1.0, help="Smoothing alpha for live frames (0.0-1.0). Lower reduces jitter; 1.0 disables smoothing.")
    parser.add_argument("--allow-expand-bounds", dest="allow_expand_bounds", action="store_true", help="Allow live view bounds to expand during run (otherwise bounds are fixed after first frame)")
    parser.add_argument("--env-clip", dest="env_clip", type=str, default=None, help="Percentile clip for env background as 'low,high' (e.g. 2,98)")
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
    if args.live:
        host = args.host
        port = args.port
        positions = np.zeros((1, 0, 2), dtype=float)
        win = MainWindow(positions, watchdog_seconds=args.watchdog_seconds, pad=args.view_pad, force_vbo=args.force_vbo, point_size=args.point_size)
        # if an env depth file was passed, install it on the replay widget
        try:
                if args.env_depth is not None:
                    try:
                        env_clip_pct = None
                        if args.env_clip:
                            try:
                                parts = [float(p) for p in args.env_clip.split(',')]
                                if len(parts) >= 2:
                                    env_clip_pct = (parts[0], parts[1])
                            except Exception:
                                env_clip_pct = None
                        rv = ReplayWidget(positions, pad=args.view_pad, point_size=args.point_size, env_depth=args.env_depth, allow_expand_bounds=args.allow_expand_bounds, smooth_alpha=args.smooth_alpha, env_clip_pct=env_clip_pct)
                        swap_viewer_in_main(win, rv)
                    except Exception:
                        pass
        except Exception:
            pass
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
                            gl_view = GLViewer(positions_live, pad=win._pad, force_vbo=win._force_vbo, point_size=win._point_size)
                        except Exception:
                            # if GLViewer init fails, log and continue to fallback
                            log.exception('GLViewer init failed; falling back to other renderers')
                            gl_view = None
                        if gl_view is not None:
                            try:
                                swap_viewer_in_main(win, gl_view)
                                # apply window pan/zoom state
                                try:
                                    gl_view._pad = win._pad
                                    gl_view._pan_x = win._pan_x
                                    gl_view._pan_y = win._pan_y
                                except Exception:
                                    pass
                            except Exception:
                                logger.exception('Failed to swap GLViewer into MainWindow')
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
                                swap_viewer_in_main(win, pg_view)
                            except Exception:
                                logger.exception('Failed to swap PGViewer into MainWindow')
                        except Exception:
                            logger.exception('pyqtgraph renderer init failed')
                    # set positions on whichever viewer we have (diagnostic: prefer ReplayWidget)
                    try:
                        # Prefer the reliable software `ReplayWidget` renderer unless GL requested
                        if not args.use_gl and not isinstance(win.viewer, ReplayWidget):
                            replay = ReplayWidget(positions_live, pad=win._pad, point_size=win._point_size)
                            try:
                                swap_viewer_in_main(win, replay)
                            except Exception:
                                # fallback: insert into layout and remove old viewer
                                parent = win.centralWidget()
                                layout = parent.layout()
                                try:
                                    idx = layout.indexOf(win.viewer)
                                except Exception:
                                    idx = -1
                                if idx is None or idx < 0:
                                    try:
                                        layout.addWidget(replay)
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        layout.insertWidget(idx, replay)
                                    except Exception:
                                        try:
                                            layout.addWidget(replay)
                                        except Exception:
                                            pass
                                    try:
                                        layout.removeWidget(win.viewer)
                                    except Exception:
                                        pass
                                try:
                                    win.viewer.setParent(None)
                                except Exception:
                                    pass
                                win.viewer = replay
                            # ensure the widget is visible and repaint
                            try:
                                win.viewer.show()
                                win.viewer.repaint()
                                log.debug('Swapped in ReplayWidget for live frames')
                            except Exception:
                                log.exception('Failed to show/repaint ReplayWidget')
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
                                try:
                                    log.debug('env bbox=%s', getattr(win.viewer, '_env_bbox', None))
                                except Exception:
                                    pass
                        except Exception:
                            log.exception('Failed computing live frame bounds')

                            # prefer viewer's update_frame if available for live updates
                            try:
                                # increment live counter and maintain trail buffer for smoother motion
                                try:
                                    if hasattr(win.viewer, '_live_count'):
                                        win.viewer._live_count = getattr(win.viewer, '_live_count', 0) + 1
                                    if hasattr(win.viewer, '_trail_buf'):
                                        try:
                                            win.viewer._trail_buf.append(arr.copy())
                                            if len(win.viewer._trail_buf) > getattr(win.viewer, '_trail_length', 8):
                                                win.viewer._trail_buf.pop(0)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                if hasattr(win.viewer, 'update_frame'):
                                    win.viewer.update_frame(arr)
                                else:
                                    win.viewer.positions = positions_live
                            except Exception:
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
                                try:
                                    # log a small sample of agent coordinates (first 5) for numeric comparison
                                    sample = positions_live[0, :5, :].tolist()
                                    log.debug('Sample agent coords (first 5): %s', sample)
                                except Exception:
                                    pass
                        except Exception:
                            log.exception('Failed to compute live bounds for logging')
                        # diagnostic flag disabled by default; do not force large points
                        # if legacy ReplayWidget
                        if hasattr(win.viewer, 'T') and not hasattr(win.viewer, 'update_frame'):
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

                            # ensure viewer inherits window pad/pan state
                            try:
                                win.viewer._pad = win._pad
                                win.viewer._pan_x = win._pan_x
                                win.viewer._pan_y = win._pan_y
                            except Exception:
                                pass

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
                    swap_viewer_in_main(win, pg_view)
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

    # Try to load environment depth from the HDF5 file
    env_depth_array, env_x_coords, env_y_coords = None, None, None
    battery_array = None
    heading_array = None
    if path.endswith('.h5') or path.endswith('.hdf5'):
        try:
            depth, x_coords, y_coords = load_env_from_h5(path)
            if depth is not None:
                env_depth_array = depth
                env_x_coords = x_coords
                env_y_coords = y_coords
                print(f"Loaded environment from HDF5: depth shape {depth.shape}")
        except Exception as ex:
            print(f"Could not load environment from HDF5: {ex}")
        
        # Try to load battery data for fatigue visualization
        try:
            battery_array = load_battery_from_h5(path)
            if battery_array is not None:
                print(f"Loaded battery data from HDF5: shape {battery_array.shape}")
        except Exception as ex:
            print(f"Could not load battery data from HDF5: {ex}")
        
        # Try to load heading data for oriented fish rendering
        try:
            heading_array = load_heading_from_h5(path)
            if heading_array is not None:
                print(f"Loaded heading data from HDF5: shape {heading_array.shape}")
        except Exception as ex:
            print(f"Could not load heading from HDF5: {ex}")

    # create window and pick renderer based on agent count
    win = MainWindow(positions, pad=args.view_pad, force_vbo=args.force_vbo, point_size=args.point_size)
    
    # Determine agent count for renderer selection
    try:
        Nagents = positions.shape[1]
    except Exception:
        Nagents = 0
    
    # Use ReplayWidget with fish bodies for better visualization (up to 5000 agents)
    # For very large simulations, fallback to GLViewer
    if Nagents < 5001:
        try:
            env_clip_pct = None
            if args.env_clip:
                try:
                    parts = [float(p) for p in args.env_clip.split(',')]
                    if len(parts) >= 2:
                        env_clip_pct = (parts[0], parts[1])
                except Exception:
                    env_clip_pct = None
            # Pass depth arrays if loaded from HDF5
            rv = ReplayWidget(
                positions, 
                pad=args.view_pad, 
                point_size=args.point_size, 
                env_depth=args.env_depth, 
                allow_expand_bounds=args.allow_expand_bounds, 
                smooth_alpha=args.smooth_alpha, 
                env_clip_pct=env_clip_pct,
                env_depth_array=env_depth_array,
                env_x_coords=env_x_coords,
                env_y_coords=env_y_coords,
                battery_array=battery_array,
                heading_array=heading_array
            )
            swap_viewer_in_main(win, rv)
        except Exception as ex:
            logger.exception('Failed to create ReplayWidget')
    else:
        # Fallback to GLViewer for very large simulations (5001+ agents)
        use_gl_mode = args.use_gl or (Nagents >= 5001)
        if use_gl_mode:
            try:
                glw = GLViewer(positions, pad=win._pad, force_vbo=win._force_vbo, point_size=win._point_size, env_depth_array=env_depth_array, env_x_coords=env_x_coords, env_y_coords=env_y_coords, battery_array=battery_array, heading_array=heading_array)
                try:
                    swap_viewer_in_main(win, glw)
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
