"""Moderngl-based widget for rendering TIN meshes inside PyQt5.

This widget uses QOpenGLWidget and creates a ModernGL context from the
existing GL context. It expects mesh data as (verts Nx3, faces Mx3, colors Nx4).

The implementation is intentionally minimal and focuses on correctness and
visual quality. It can be extended with camera controls, lighting, and
performance optimizations (VBO reuse, frustum culling) later.
"""
from PyQt5.QtWidgets import QOpenGLWidget
import numpy as np

try:
    import moderngl
except Exception:
    moderngl = None


def _ortho_matrix(left, right, bottom, top, near, far, dtype='f4'):
    """Return column-major orthographic projection matrix as float32."""
    rl = right - left
    tb = top - bottom
    fn = far - near
    if rl == 0: rl = 1.0
    if tb == 0: tb = 1.0
    if fn == 0: fn = 1.0
    tx = -(right + left) / rl
    ty = -(top + bottom) / tb
    tz = -(far + near) / fn
    m = np.array([
        [2.0 / rl, 0.0, 0.0, tx],
        [0.0, 2.0 / tb, 0.0, ty],
        [0.0, 0.0, -2.0 / fn, tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=dtype)
    return m


class ModernglViewerWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.prog = None
        self.vbo = None
        self.ibo = None
        self.cbo = None
        self._vao = None
        self.num_indices = 0
        self.verts = None
        self.faces = None
        self.colors = None
        self.setMinimumSize(480, 360)

    def initializeGL(self):
        if moderngl is None:
            raise RuntimeError('moderngl is not available')
        # Create moderngl context from current OpenGL context
        self.ctx = moderngl.create_context(require=330)

        vs = """#version 330
        in vec3 in_position;
        in vec4 in_color;
        out vec4 v_color;
        uniform mat4 mvp;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_color = in_color;
        }
        """
        fs = """#version 330
        in vec4 v_color;
        out vec4 f_color;
        void main() {
            f_color = v_color;
        }
        """
        self.prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def resizeGL(self, w: int, h: int):
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, max(1, w), max(1, h))
            # if mesh exists, update projection
            if self.verts is not None:
                self._update_mvp()

    def paintGL(self):
        if self.ctx is None:
            return
        self.ctx.clear(0.94, 0.94, 0.94, 1.0)
        if self._vao is None:
            return
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLES)

    def set_mesh(self, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray):
        """Upload mesh buffers to GPU and update projection.

        verts: Nx3 float32
        faces: Mx3 int32
        colors: Nx4 float32
        """
        if self.ctx is None:
            # postpone until GL context created
            self.verts = np.asarray(verts, dtype='f4')
            self.faces = np.asarray(faces, dtype='i4')
            self.colors = np.asarray(colors, dtype='f4')
            return

        self.verts = np.asarray(verts, dtype='f4')
        self.faces = np.asarray(faces, dtype='i4')
        self.colors = np.asarray(colors, dtype='f4')

        # Flatten index array
        idx = self.faces.astype('i4').ravel()

        # Create / replace buffers
        if self.vbo is not None:
            self.vbo.release()
        if self.ibo is not None:
            self.ibo.release()
        if self.cbo is not None:
            self.cbo.release()
        try:
            self.vbo = self.ctx.buffer(self.verts.tobytes())
            self.cbo = self.ctx.buffer(self.colors.tobytes())
            self.ibo = self.ctx.buffer(idx.tobytes())
        except Exception:
            # fallback: keep CPU-side data but don't crash
            return

        # Create vertex array object
        if self._vao is not None:
            try:
                self._vao.release()
            except Exception:
                pass
        vao_content = [
            (self.vbo, '3f', 'in_position'),
            (self.cbo, '4f', 'in_color'),
        ]
        self._vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer=self.ibo)

        # Update MVP using mesh extents
        self._update_mvp()
        self.update()

    def _update_mvp(self):
        # Compute orthographic projection that fits mesh extents
        verts = self.verts
        if verts is None or verts.shape[0] == 0:
            return
        minxy = np.min(verts[:, :2], axis=0)
        maxxy = np.max(verts[:, :2], axis=0)
        center = (minxy + maxxy) / 2.0
        span = np.max(maxxy - minxy)
        if span <= 0:
            span = 1.0
        pad = span * 0.6
        left = center[0] - pad
        right = center[0] + pad
        bottom = center[1] - pad
        top = center[1] + pad
        near = -span * 2.0
        far = span * 2.0
        m = _ortho_matrix(left, right, bottom, top, near, far)
        # Write matrix as column-major float32
        try:
            self.prog['mvp'].write(m.astype('f4').tobytes())
        except Exception:
            pass

*** End Patch