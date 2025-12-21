"""CPU fallback renderer for environments where GL presentation fails.

This widget provides the same minimal public API used by the shim:
- `set_mesh(verts, faces, colors)`
- `set_heightmap(...)` (delegates to mesh creator)
- signals: `fbo_preview_ready` (QImage) and `presentation_ok` (bool)

Rendering is done on the CPU using Pillow to rasterize triangles into an
RGBA image. It is intended as a reliable visual fallback on systems where
GL compositing fails (e.g., compositor/ANGLE issues on some Windows setups).
"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from PyQt5 import QtGui
import numpy as np
try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None


class CPUViewerWidget(QWidget):
    fbo_preview_ready = pyqtSignal(object)  # QImage
    presentation_ok = pyqtSignal(bool)

    def __init__(self, parent=None, refresh_hz: int = 20):
        super().__init__(parent)
        self.verts = None
        self.faces = None
        self.colors = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(int(1000 / max(1, refresh_hz)))
        self.setMinimumSize(480, 360)

    def size_in_pixels(self):
        w, h = max(1, self.width()), max(1, self.height())
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            try:
                dpr = float(self.devicePixelRatio())
            except Exception:
                dpr = 1.0
        return max(1, int(round(w * dpr))), max(1, int(round(h * dpr)))

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray):
        self.verts = np.asarray(verts, dtype='f4')
        self.faces = np.asarray(faces, dtype='i4')
        self.colors = np.asarray(colors, dtype='f4')

    def set_heightmap(self, *args, **kwargs):
        # delegate to higher-level mesh builder in shim; shim uses its mesh_builder
        return

    def _on_tick(self):
        # render current mesh to QImage and emit preview
        try:
            w_px, h_px = self.size_in_pixels()
            if Image is None or self.verts is None or self.faces is None:
                # emit a blank image to indicate presence
                qimg = QtGui.QImage(w_px, h_px, QtGui.QImage.Format_RGBA8888)
                qimg.fill(Qt.transparent)
                try:
                    self.fbo_preview_ready.emit(qimg)
                    self.presentation_ok.emit(True)
                except Exception:
                    pass
                return

            # prepare white background with slight neutral tone
            pil = Image.new('RGBA', (w_px, h_px), (20, 20, 30, 255))
            draw = ImageDraw.Draw(pil, 'RGBA')

            # compute 2D projection: fit verts XY into image
            xs = self.verts[:, 0]
            ys = self.verts[:, 1]
            minx, maxx = float(np.min(xs)), float(np.max(xs))
            miny, maxy = float(np.min(ys)), float(np.max(ys))
            if maxx - minx == 0:
                maxx = minx + 1.0
            if maxy - miny == 0:
                maxy = miny + 1.0

            def proj(x, y):
                # map to pixel coords with padding
                px = (x - minx) / (maxx - minx)
                py = (y - miny) / (maxy - miny)
                ix = int(px * (w_px - 1))
                iy = int((1.0 - py) * (h_px - 1))
                return ix, iy

            # simple painter's algorithm: sort triangles by average z
            tris = self.faces.reshape(-1, 3)
            tri_z = np.mean(self.verts[tris][:, :, 2], axis=1)
            order = np.argsort(tri_z)  # back-to-front

            for ti in order:
                tri = tris[ti]
                pts = [proj(*self.verts[v][:2]) for v in tri]
                cols = (self.colors[tri][:, :3] * 255).astype(int)
                avg = tuple(np.mean(cols, axis=0).astype(int).tolist())
                color = (avg[0], avg[1], avg[2], 255)
                try:
                    draw.polygon(pts, fill=color)
                except Exception:
                    # fallback: draw lines
                    draw.line(pts + [pts[0]], fill=color, width=1)

            # convert PIL to QImage
            data = pil.tobytes('raw', 'RGBA')
            qimg = QtGui.QImage(data, w_px, h_px, QtGui.QImage.Format_RGBA8888)
            try:
                self.fbo_preview_ready.emit(qimg)
                self.presentation_ok.emit(True)
            except Exception:
                pass
        except Exception:
            try:
                qimg = QtGui.QImage(1, 1, QtGui.QImage.Format_RGBA8888)
                qimg.fill(Qt.transparent)
                self.fbo_preview_ready.emit(qimg)
                self.presentation_ok.emit(False)
            except Exception:
                pass

    # expose minimal API used by shim
    def set_agents(self, positions, colors=None, size: float = 4.0):
        # no-op for now
        pass
