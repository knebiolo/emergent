import sys
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl

app = QtWidgets.QApplication([])
view = gl.GLViewWidget()
view.show()
view.setWindowTitle('Minimal OpenGL Test')
view.setGeometry(100, 100, 800, 600)

# Add a simple cube
import numpy as np
verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
])
faces = np.array([
    [0, 1, 2], [0, 2, 3],
    [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5],
    [3, 0, 4], [3, 4, 7],
])
colors = np.array([[255, 0, 0, 255]] * len(verts), dtype=np.ubyte)
meshdata = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
mesh = gl.GLMeshItem(meshdata=meshdata, smooth=False, drawEdges=True, shader='shaded', glOptions='opaque')
view.addItem(mesh)

sys.exit(app.exec_())
