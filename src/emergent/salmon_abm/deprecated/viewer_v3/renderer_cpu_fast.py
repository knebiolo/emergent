"""Fast CPU rasterizer using NumPy.

Produces RGBA images as QImage-compatible bytes. This is a compact,
optimized (NumPy) triangle rasterizer with a Z-buffer and per-vertex
color interpolation. It is intended as a reliable fallback renderer.
"""
from typing import Tuple
import numpy as np
from PyQt5 import QtGui
from PIL import Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QTimer, Qt


def ndc_to_screen(ndc, width, height):
    # ndc: (..., 3) with x,y in -1..1, z in clip space
    x = (ndc[..., 0] * 0.5 + 0.5) * (width - 1)
    y = (1.0 - (ndc[..., 1] * 0.5 + 0.5)) * (height - 1)
    z = ndc[..., 2]
    return np.stack([x, y, z], axis=-1)


def edge_function(a, b, c):
    return (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) - (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0])


def rasterize_triangles(width: int, height: int, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> QtGui.QImage:
    """Rasterize triangles to a QImage.

    verts: (N, 3) positions in NDC space (-1..1)
    faces: (M, 3) indices
    colors: (N, 3) vertex colors 0..1
    """
    # Prepare buffers
    color_buf = np.zeros((height, width, 4), dtype=np.uint8)
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)

    # Convert verts to screen
    screen = ndc_to_screen(verts, width, height)

    for face in faces:
        v0, v1, v2 = screen[face[0]], screen[face[1]], screen[face[2]]
        c0, c1, c2 = colors[face[0]], colors[face[1]], colors[face[2]]

        # Compute tri bbox
        min_x = max(int(np.floor(min(v0[0], v1[0], v2[0]))), 0)
        max_x = min(int(np.ceil(max(v0[0], v1[0], v2[0]))), width - 1)
        min_y = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
        max_y = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), height - 1)

        if min_x > max_x or min_y > max_y:
            continue

        # create grid of sample points
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx, gy], axis=-1).astype(np.float32)

        # compute barycentrics via edge functions
        area = edge_function(v0, v1, v2)
        if np.isclose(area, 0.0):
            continue

        w0 = edge_function(v1, v2, pts)
        w1 = edge_function(v2, v0, pts)
        w2 = edge_function(v0, v1, pts)

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside):
            continue

        w0 = w0[inside] / area
        w1 = w1[inside] / area
        w2 = w2[inside] / area

        pts_inside = pts[inside]

        # Interpolate depth and color
        z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
        cols = (w0[:, None] * c0 + w1[:, None] * c1 + w2[:, None] * c2)

        # Write to buffers
        py = pts_inside[:, 1].astype(np.intp)
        px = pts_inside[:, 0].astype(np.intp)
        for i in range(len(px)):
            xi, yi = px[i], py[i]
            zi = z[i]
            if zi < depth_buf[yi, xi]:
                depth_buf[yi, xi] = zi
                color_buf[yi, xi, :3] = np.clip((cols[i] * 255.0), 0, 255).astype(np.uint8)
                color_buf[yi, xi, 3] = 255

    # Convert to QImage
    h, w = color_buf.shape[:2]
    qimg = QtGui.QImage(color_buf.data.tobytes(), w, h, QtGui.QImage.Format_RGBA8888)
    return qimg


if __name__ == '__main__':
    # headless sanity test: render a single colored tetrahedron
    w, h = 512, 384
    verts = np.array([
        [-0.6, -0.6,  0.2],
        [ 0.6, -0.6,  0.4],
        [ 0.0,  0.6,  0.6],
        [ 0.0,  0.0, -0.2],
    ], dtype='f4')
    faces = np.array([
        [0,1,2],
        [0,1,3],
        [1,2,3],
        [2,0,3],
    ], dtype='i4')
    colors = np.array([
        [1,0,0], [0,1,0], [0,0,1], [1,1,0]
    ], dtype='f4')

    qimg = rasterize_triangles(w, h, verts, faces, colors)
    # save via PIL for easy inspection
    buf = qimg.bits().asstring(w*h*4)
    img = Image.frombytes('RGBA', (w,h), buf)
    outp = 'outputs/cpu_raster_snapshot.png'
    img.save(outp)
    print('Wrote', outp)


class FastCPUViewerWidget(QWidget):
    fbo_preview_ready = pyqtSignal(object)  # QImage
    presentation_ok = pyqtSignal(bool)

    def __init__(self, parent=None, refresh_hz: int = 20):
        super().__init__(parent)
        self.verts = None
        self.faces = None
        self.colors = None
        self._agents = None
        self._agent_colors = None
        self._agent_size = 4.0
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

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, vert_exag: float = 1.0):
        """Accept world-space vertices and project to NDC for rasterization.

        - `verts`: (N,3) world coords (x,y,z)
        - `faces`: (M,3) indices
        - `colors`: (N,3) or (N,4) colors in 0..1
        - `vert_exag`: vertical exaggeration multiplier applied to Z before projection
        """
        verts = np.asarray(verts, dtype='f4')
        faces = np.asarray(faces, dtype='i4')
        colors = np.asarray(colors, dtype='f4')

        if verts.size == 0 or faces.size == 0:
            self.verts = verts
            self.faces = faces
            self.colors = colors
            return

        # apply vertical exaggeration
        verts_proj = verts.copy()
        verts_proj[:, 2] = verts_proj[:, 2] * float(vert_exag)

        # compute XY bounds and center/scale to fit into NDC [-1,1]
        xs = verts_proj[:, 0]
        ys = verts_proj[:, 1]
        minx, maxx = float(np.min(xs)), float(np.max(xs))
        miny, maxy = float(np.min(ys)), float(np.max(ys))
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)
        spanx = maxx - minx
        spany = maxy - miny
        span = max(spanx, spany)
        if span == 0:
            span = 1.0

        # map world XY to NDC [-0.9, 0.9] leaving a margin
        margin = 0.9
        ndc_x = (verts_proj[:, 0] - cx) / span * (2.0 * margin)
        ndc_y = (verts_proj[:, 1] - cy) / span * (2.0 * margin)

        # Normalize z into [0,1] based on min/max Z for depth sorting, then remap to -1..1 clip
        zs = verts_proj[:, 2]
        zmin, zmax = float(np.min(zs)), float(np.max(zs))
        if np.isclose(zmax, zmin):
            ndc_z = np.full_like(zs, 0.5)
        else:
            ndc_z = (zs - zmin) / (zmax - zmin)

        # final NDC in range [-1,1] for x/y, z in 0..1 (depth)
        ndc = np.stack([ndc_x, ndc_y, ndc_z], axis=-1)

        self.verts = ndc
        self.faces = faces
        # ensure colors have 3 channels
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        self.colors = np.clip(colors, 0.0, 1.0)

    def set_heightmap(self, *args, **kwargs):
        return

    def _on_tick(self):
        try:
            w_px, h_px = self.size_in_pixels()
            if self.verts is None or self.faces is None or self.colors is None:
                qimg = QtGui.QImage(w_px, h_px, QtGui.QImage.Format_RGBA8888)
                qimg.fill(Qt.transparent)
                try:
                    self.fbo_preview_ready.emit(qimg)
                    self.presentation_ok.emit(True)
                except Exception:
                    pass
                return

            qimg = rasterize_triangles(w_px, h_px, self.verts, self.faces, self.colors)
            # If agents present, draw them on top by converting to array, blitting, and rewrapping
            try:
                if getattr(self, '_agents', None) is not None:
                    # access bits and manipulate the buffer
                    ptr = qimg.bits().asstring(w_px * h_px * 4)
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h_px, w_px, 4)).copy()
                    # draw agents: positions expected in NDC [-1,1] or pixel coords
                    ag = np.asarray(self._agents, dtype=float)
                    if ag.size != 0:
                        # If ag shape is (N,2) assume world XY mapped to NDC similarly to verts
                        if ag.shape[1] == 2:
                            # map NDC XY to screen pixels using same ndc_to_screen mapping but without z
                            px = ((ag[:, 0] * 0.5 + 0.5) * (w_px - 1)).astype(np.intp)
                            py = ((1.0 - (ag[:, 1] * 0.5 + 0.5)) * (h_px - 1)).astype(np.intp)
                        else:
                            # if given in pixel coords
                            px = ag[:, 0].astype(np.intp)
                            py = ag[:, 1].astype(np.intp)
                        cols = None
                        if getattr(self, '_agent_colors', None) is not None:
                            cols = np.asarray(self._agent_colors, dtype=np.uint8)
                        for i in range(len(px)):
                            xi = np.clip(px[i], 0, w_px - 1)
                            yi = np.clip(py[i], 0, h_px - 1)
                            col = cols[i] if (cols is not None and i < len(cols)) else np.array([255, 255, 0], dtype=np.uint8)
                            # draw a small filled square as agent marker
                            r = max(1, int(round(self._agent_size)))
                            x0 = max(0, xi - r)
                            x1 = min(w_px - 1, xi + r)
                            y0 = max(0, yi - r)
                            y1 = min(h_px - 1, yi + r)
                            arr[y0:y1+1, x0:x1+1, 0:3] = col[None, None, :]
                            arr[y0:y1+1, x0:x1+1, 3] = 255
                    # create new QImage from modified array
                    qimg = QtGui.QImage(arr.data.tobytes(), w_px, h_px, QtGui.QImage.Format_RGBA8888)
            except Exception:
                pass
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

    def set_agents(self, positions, colors=None, size: float = 4.0):
        try:
            arr = np.asarray(positions, dtype=float)
            self._agents = arr
            if colors is not None:
                c = np.asarray(colors, dtype=float)
                # Convert 0..1 to 0..255 if needed
                if c.dtype == float or c.max() <= 1.0:
                    c = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    c = c.astype(np.uint8)
                self._agent_colors = c
            else:
                self._agent_colors = None
            self._agent_size = float(size)
        except Exception:
            self._agents = None
            self._agent_colors = None
            self._agent_size = float(size)
