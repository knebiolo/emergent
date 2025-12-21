"""GPU-backed viewer using pyqtgraph.opengl.

Provides `FastGLViewerWidget` which tries to use `pyqtgraph.opengl` to draw
the mesh as a `GLMeshItem` and agents as a `GLScatterPlotItem`.
"""
from typing import Optional
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except Exception:
    pg = None
    gl = None


class FastGLViewerWidget(QtWidgets.QWidget):
    fbo_preview_ready = pyqtSignal(object)
    presentation_ok = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        if gl is None:
            raise RuntimeError('pyqtgraph.opengl not available')

        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 6
        try:
            self.view.setCameraPosition(distance=6, elevation=20)
        except Exception:
            pass
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.view)

        self.mesh_item = None
        self.scatter = None
        self.setMinimumSize(480, 360)
        sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sp)

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, vert_exag: float = 1.0):
        try:
            # pyqtgraph GLMeshItem expects faces as Nx3 int32 and vertex colors per-vertex optional
            verts = np.asarray(verts, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int32)
            colors = np.asarray(colors, dtype=np.float32)
            # flatten colors to per-vertex RGBA
            if colors.shape[1] == 3:
                rgba = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.float32)], axis=1)
            else:
                rgba = colors

            # build meshdata
            md = gl.MeshData(vertexes=verts[:, :3], faces=faces)
            if self.mesh_item is None:
                self.mesh_item = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=True, drawEdges=False)
                self.view.addItem(self.mesh_item)
            else:
                self.mesh_item.setMeshData(meshdata=md)
        except Exception as e:
            raise

    def set_agents(self, positions, colors=None, size: float = 4.0):
        try:
            pos = np.asarray(positions, dtype=np.float32)
            # expand to 3D by adding zero z
            if pos.ndim == 2 and pos.shape[1] == 2:
                pos3 = np.zeros((pos.shape[0], 3), dtype=np.float32)
                pos3[:, 0:2] = pos
            else:
                pos3 = pos
            col_array = None
            if colors is not None:
                c = np.asarray(colors)
                if c.dtype == float or c.max() <= 1.0:
                    c = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.ubyte)
                # ensure shape Nx3 or Nx4
                if c.ndim == 1:
                    c = np.tile(c, (pos3.shape[0], 1))
                if c.shape[1] == 3:
                    a = np.full((c.shape[0], 1), 255, dtype=np.ubyte)
                    c = np.concatenate([c.astype(np.ubyte), a], axis=1)
                col_array = c

            if self.scatter is None:
                self.scatter = gl.GLScatterPlotItem(pos=pos3, size=float(size), pxMode=True, color=col_array)
                self.view.addItem(self.scatter)
            else:
                try:
                    self.scatter.setData(pos=pos3, size=float(size), color=col_array)
                except Exception:
                    # fallback: only set positions
                    self.scatter.setData(pos=pos3, size=float(size))
        except Exception:
            pass
