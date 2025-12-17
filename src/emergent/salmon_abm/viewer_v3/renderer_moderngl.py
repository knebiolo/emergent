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
try:
    import matplotlib.cm as cm
except Exception:
    cm = None


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
        # render agents if present
        try:
            if getattr(self, '_agent_vao', None) is not None:
                self.ctx.disable(moderngl.DEPTH_TEST)
                self._agent_vao.render(moderngl.POINTS)
                self.ctx.enable(moderngl.DEPTH_TEST)
        except Exception:
            pass

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

    def set_heightmap(self, depth_grid: np.ndarray, bbox: tuple | None = None, max_res: int = 256, colormap: str = 'viridis', vert_exag: float = 1.0):
        """Create a regular-grid mesh from a 2D depth raster and upload to GPU.

        Args:
            depth_grid: HxW 2D array of depths.
            bbox: (minx, miny, maxx, maxy) in world coordinates for the grid. If None, uses unit coords.
            max_res: maximum grid dimension (either axis) to downsample to for performance.
            colormap: matplotlib colormap name for coloring.
            vert_exag: vertical exaggeration multiplier for Z values.
        """
        if depth_grid is None or depth_grid.size == 0:
            return
        arr = np.asarray(depth_grid, dtype=float)
        h, w = arr.shape
        # downsample to max_res for max dimension
        scale = max(1, int(max(h, w) / max_res))
        if scale > 1:
            arr = arr[::scale, ::scale]
            h, w = arr.shape

        if bbox is None:
            minx, miny, maxx, maxy = 0.0, 0.0, float(w - 1), float(h - 1)
        else:
            minx, miny, maxx, maxy = bbox

        xs = np.linspace(minx, maxx, w, dtype=float)
        ys = np.linspace(miny, maxy, h, dtype=float)
        xv, yv = np.meshgrid(xs, ys)

        zs = np.nan_to_num(arr, nan=0.0) * float(vert_exag)

        # create vertices (flattened)
        verts = np.column_stack([xv.ravel().astype('f4'), yv.ravel().astype('f4'), zs.ravel().astype('f4')])

        # create faces (two triangles per grid cell)
        # indices: (i,j) -> idx = i*w + j
        idxs = []
        for i in range(h - 1):
            for j in range(w - 1):
                a = i * w + j
                b = a + 1
                c = a + w
                d = c + 1
                # triangle 1: a, b, d
                idxs.append((a, b, d))
                # triangle 2: a, d, c
                idxs.append((a, d, c))
        faces = np.array(idxs, dtype='i4') if len(idxs) > 0 else np.zeros((0, 3), dtype='i4')

        # color mapping
        if cm is not None:
            try:
                cmap = cm.get_cmap(colormap)
                vmin = float(np.nanmin(zs))
                vmax = float(np.nanmax(zs))
                denom = vmax - vmin if (vmax - vmin) != 0 else 1.0
                normed = ((zs.ravel() - vmin) / denom).clip(0.0, 1.0)
                rgba = cmap(normed)
                colors = np.asarray(rgba, dtype='f4')
            except Exception:
                colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype='f4'), (verts.shape[0], 1))
        else:
            colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype='f4'), (verts.shape[0], 1))

        # reuse existing set_mesh upload path
        self.set_mesh(verts, faces, colors)

    def set_agents(self, positions: np.ndarray, colors: np.ndarray | None = None, size: float = 4.0):
        """Upload agent positions (Nx3) and optional colors (Nx4) as GL points.

        positions: Nx3 float32
        colors: Nx4 float32 or None
        """
        if self.ctx is None:
            # store for later
            self._pending_agents = (np.asarray(positions, dtype='f4'), None if colors is None else np.asarray(colors, dtype='f4'))
            return
        pos = np.asarray(positions, dtype='f4')
        if colors is None:
            cols = np.tile(np.array([1.0, 0.2, 0.2, 1.0], dtype='f4'), (pos.shape[0], 1))
        else:
            cols = np.asarray(colors, dtype='f4')

        # release old buffers
        try:
            if getattr(self, '_agent_vbo', None) is not None:
                self._agent_vbo.release()
            if getattr(self, '_agent_cbo', None) is not None:
                self._agent_cbo.release()
            if getattr(self, '_agent_vao', None) is not None:
                try:
                    self._agent_vao.release()
                except Exception:
                    pass
        except Exception:
            pass

        self._agent_vbo = self.ctx.buffer(pos.tobytes())
        self._agent_cbo = self.ctx.buffer(cols.tobytes())
        # create a simple vao reusing the same program inputs
        try:
            self._agent_vao = self.ctx.vertex_array(self.prog, [(self._agent_vbo, '3f', 'in_position'), (self._agent_cbo, '4f', 'in_color')])
        except Exception:
            # fallback: create minimal vao mapping
            self._agent_vao = None
        # set gl point size via program uniform if available (not in this simple shader)
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