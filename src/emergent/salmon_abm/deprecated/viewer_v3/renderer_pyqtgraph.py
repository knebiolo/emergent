"""Pyqtgraph-backed viewer wrapper that uses the fast CPU rasterizer for drawing.

This provides a thin adapter so the shim can prefer `FastPyqtgraphViewerWidget`
when `pyqtgraph` is available, but still render via the NumPy rasterizer.
"""
from typing import Optional
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal, QTimer

try:
    import pyqtgraph as pg
except Exception:
    pg = None

from emergent.salmon_abm.viewer_v3.renderer_cpu_fast import FastCPUViewerWidget, rasterize_triangles


class FastPyqtgraphViewerWidget(QtWidgets.QWidget):
    """Adapter: expose same signals and API as `FastCPUViewerWidget` but
    integrate into a pyqtgraph layout if available.
    """
    fbo_preview_ready = pyqtSignal(object)
    presentation_ok = pyqtSignal(bool)

    def __init__(self, parent=None, refresh_hz: int = 20):
        super().__init__(parent)
        self._cpu_widget = FastCPUViewerWidget(self, refresh_hz=refresh_hz)
        # forward signals
        try:
            self._cpu_widget.fbo_preview_ready.connect(self.fbo_preview_ready.emit)
            self._cpu_widget.presentation_ok.connect(self.presentation_ok.emit)
            # also update the pyqtgraph ImageItem live
            try:
                self._cpu_widget.fbo_preview_ready.connect(self.update_preview_from_qimage)
            except Exception:
                pass
        except Exception:
            pass

        # layout: if pyqtgraph is available, embed a simple ImageItem for fast blit
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        if pg is not None:
            try:
                self._view = pg.GraphicsLayoutWidget()
                self._plot = self._view.addViewBox()
                self._img_item = pg.ImageItem()
                self._plot.addItem(self._img_item)
                self._plot.setAspectLocked(True)
                self._layout.addWidget(self._view)
            except Exception:
                self._view = None
                self._img_item = None
                # fallback: just show the CPU widget itself
                self._layout.addWidget(self._cpu_widget)
        else:
            self._view = None
            self._img_item = None
            self._layout.addWidget(self._cpu_widget)

        self.setMinimumSize(480, 360)

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, vert_exag: float = 1.0):
        # delegate to CPU widget which computes NDC mapping
        self._cpu_widget.set_mesh(verts, faces, colors, vert_exag=vert_exag)

    def set_heightmap(self, *args, **kwargs):
        return self._cpu_widget.set_heightmap(*args, **kwargs)

    def set_agents(self, positions, colors=None, size: float = 4.0):
        # forward to CPU widget; CPU widget will rasterize agents as points
        try:
            return self._cpu_widget.set_agents(positions, colors=colors, size=size)
        except Exception:
            return None

    def size_in_pixels(self):
        return self._cpu_widget.size_in_pixels()

    # expose a manual refresh that updates the ImageItem from the latest raster
    def update_preview_from_qimage(self, qimg: QtGui.QImage):
        if qimg is None:
            return
        try:
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits().asstring(w * h * 4)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
            # pyqtgraph expects shape (rows, cols) or (rows, cols, 3); drop alpha
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            if self._img_item is not None:
                try:
                    # pyqtgraph wants (cols, rows) transpose depending on origin; ImageItem handles it
                    self._img_item.setImage(rgb, autoLevels=False)
                except Exception:
                    pass
        except Exception:
            pass

    # forward timer tick from CPU widget by connecting signals externally when desired
