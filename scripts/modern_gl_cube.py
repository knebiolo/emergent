"""Small ModernGL + PyQt5 demo that renders a rotating cube.

Run with: python scripts/modern_gl_cube.py
If ModernGL is not installed the script will display a message and exit.
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QSurfaceFormat

try:
    import moderngl
    from PyQt5.QtWidgets import QOpenGLWidget
except Exception as e:
    print('ModernGL or PyQt5 not available:', e)
    sys.exit(1)

import numpy as np


VERTEX_SHADER = '''#version 330
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
uniform mat4 mvp;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
'''

FRAGMENT_SHADER = '''#version 330
in vec3 v_color;
out vec4 f_color;
void main() { f_color = vec4(v_color,1.0); }
'''


class CubeWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)
        self.ctx = None
        self.program = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.angle = 0.0

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

        # Cube vertices (pos + color)
        verts = np.array([
            # positions       colors
            -1,-1,-1,  1,0,0,
             1,-1,-1,  0,1,0,
             1, 1,-1,  0,0,1,
            -1, 1,-1,  1,1,0,
            -1,-1, 1,  1,0,1,
             1,-1, 1,  0,1,1,
             1, 1, 1,  1,1,1,
            -1, 1, 1,  0,0,0,
        ], dtype='f4')

        idx = np.array([
            0,1,2,  2,3,0,
            4,5,6,  6,7,4,
            0,1,5,  5,4,0,
            2,3,7,  7,6,2,
            1,2,6,  6,5,1,
            3,0,4,  4,7,3,
        ], dtype='i4')

        self.vbo = self.ctx.buffer(verts.tobytes())
        self.ibo = self.ctx.buffer(idx.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f 3f', 'in_pos', 'in_color')], index_buffer=self.ibo)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def resizeGL(self, w, h):
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, max(1, w), max(1, h))

    def paintGL(self):
        if self.ctx is None:
            return
        self.angle += 0.01
        proj = self._perspective(45.0, self.width() / max(1.0, self.height()), 0.1, 100.0)
        model = self._rotate(self.angle, (1,1,0)) @ self._translate((0,0,-6))
        mvp = proj @ model
        self.program['mvp'].write(mvp.astype('f4').tobytes())
        self.ctx.clear(0.1,0.1,0.12)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render()

    def _translate(self, t):
        x,y,z = t
        m = np.eye(4, dtype='f4')
        m[3,:3] = np.array([x,y,z], dtype='f4')
        return m

    def _rotate(self, angle, axis):
        ax = np.array(axis, dtype='f4')
        ax = ax / np.linalg.norm(ax)
        x,y,z = ax
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([
            [x*x*(1-c)+c,   x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
            [y*x*(1-c)+z*s, y*y*(1-c)+c,   y*z*(1-c)-x*s, 0],
            [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,   0],
            [0,0,0,1]
        ], dtype='f4')
        return R

    def _perspective(self, fov, aspect, near, far):
        f = 1.0/np.tan(np.deg2rad(fov)/2.0)
        M = np.zeros((4,4), dtype='f4')
        M[0,0] = f/aspect
        M[1,1] = f
        M[2,2] = (far+near)/(near-far)
        M[2,3] = (2*far*near)/(near-far)
        M[3,2] = -1.0
        return M


def main():
    app = QApplication([])
    win = QMainWindow()
    w = CubeWidget()
    win.setCentralWidget(w)
    win.resize(800,600)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
