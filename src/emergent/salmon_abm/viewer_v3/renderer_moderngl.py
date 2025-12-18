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
        # agent rendering state
        self._agent_vbo = None
        self._agent_cbo = None
        self._agent_vao = None
        self._agent_positions = None
        self._agent_colors = None
        # trails/trajectories
        self._show_trails = False
        self._trail_length = 10
        self._trajectories = None  # list/array of shape (n_agents, history, 3)
        self._show_directions = False
        self._point_size = 4.0
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
        # simple program for line/points (reuse same shaders but allow draw mode switches)

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
                # set point size via gl_PointSize in shader is not used; use built-in where available
                self._agent_vao.render(moderngl.POINTS)
                # render trails as line strips if available
                if self._show_trails and getattr(self, '_traj_vao', None) is not None:
                    try:
                        self._traj_vao.render(moderngl.LINES)
                    except Exception:
                        pass
                # optional direction arrows (rendered as lines)
                if self._show_directions and getattr(self, '_dir_vao', None) is not None:
                    try:
                        self._dir_vao.render(moderngl.LINES)
                    except Exception:
                        pass
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

    # Agent and trajectory helpers
    def set_point_size(self, size: float):
        self._point_size = float(size)
        self.update()

    def set_show_trails(self, show: bool):
        self._show_trails = bool(show)
        self.update()

    def set_trail_length(self, length: int):
        self._trail_length = int(length)
        # trim existing trajectories if necessary
        if self._trajectories is not None and self._trajectories.shape[1] > self._trail_length:
            self._trajectories = self._trajectories[:, -self._trail_length:, :]
        self._update_traj_buffers()
        self.update()

    def set_show_directions(self, show: bool):
        self._show_directions = bool(show)
        self.update()

    def _update_agent_buffers(self):
        # create or update buffers for agent positions/colors
        if self._agent_positions is None:
            return
        pos = np.asarray(self._agent_positions, dtype='f4')
        cols = np.asarray(self._agent_colors, dtype='f4') if self._agent_colors is not None else np.tile(np.array([1.0, 0.2, 0.2, 1.0], dtype='f4'), (pos.shape[0], 1))
        try:
            if getattr(self, '_agent_vbo', None) is None:
                self._agent_vbo = self.ctx.buffer(pos.tobytes())
            else:
                # partial update via write
                try:
                    self._agent_vbo.write(pos.tobytes())
                except Exception:
                    self._agent_vbo.release()
                    self._agent_vbo = self.ctx.buffer(pos.tobytes())
            if getattr(self, '_agent_cbo', None) is None:
                self._agent_cbo = self.ctx.buffer(cols.tobytes())
            else:
                try:
                    self._agent_cbo.write(cols.tobytes())
                except Exception:
                    self._agent_cbo.release()
                    self._agent_cbo = self.ctx.buffer(cols.tobytes())
            if getattr(self, '_agent_vao', None) is not None:
                try:
                    self._agent_vao.release()
                except Exception:
                    pass
            self._agent_vao = self.ctx.vertex_array(self.prog, [(self._agent_vbo, '3f', 'in_position'), (self._agent_cbo, '4f', 'in_color')])
        except Exception:
            pass

    def set_agents(self, positions: np.ndarray, colors: np.ndarray | None = None, size: float = 4.0):
        # store and update buffers
        self._agent_positions = np.asarray(positions, dtype='f4')
        self._agent_colors = None if colors is None else np.asarray(colors, dtype='f4')
        if self.ctx is None:
            return
        self._update_agent_buffers()
        # also append to trajectories history
        try:
            n = self._agent_positions.shape[0]
            if self._trajectories is None:
                self._trajectories = np.zeros((n, 1, 3), dtype='f4')
                self._trajectories[:, 0, :] = self._agent_positions
            else:
                # ensure same agent count
                if self._trajectories.shape[0] != n:
                    # reset trajectories
                    self._trajectories = np.zeros((n, 1, 3), dtype='f4')
                    self._trajectories[:, 0, :] = self._agent_positions
                else:
                    self._trajectories = np.concatenate([self._trajectories, self._agent_positions[:, None, :]], axis=1)
                    # trim
                    if self._trajectories.shape[1] > self._trail_length:
                        self._trajectories = self._trajectories[:, -self._trail_length:, :]
        except Exception:
            pass
        # update trajectory buffers for rendering
        self._update_traj_buffers()
        self.update()

    def _update_traj_buffers(self):
        # build line segments from trajectories
        try:
            if self._trajectories is None or not self._show_trails:
                # release traj vao if exists
                if getattr(self, '_traj_vao', None) is not None:
                    try:
                        self._traj_vao.release()
                    except Exception:
                        pass
                    self._traj_vao = None
                return
            # flatten trajectories into segments (pairs of points -> lines)
            n, h, _ = self._trajectories.shape
            if h < 2:
                return
            segs = []
            cols = []
            for i in range(n):
                traj = self._trajectories[i]
                for j in range(h - 1):
                    a = traj[j]
                    b = traj[j + 1]
                    segs.append(tuple(a.tolist()))
                    segs.append(tuple(b.tolist()))
                    cols.append((1.0, 0.2, 0.2, 0.8))
                    cols.append((1.0, 0.2, 0.2, 0.8))
            segs = np.array(segs, dtype='f4')
            cols = np.array(cols, dtype='f4')
            # create buffers
            try:
                if getattr(self, '_traj_vbo', None) is not None:
                    self._traj_vbo.release()
                if getattr(self, '_traj_cbo', None) is not None:
                    self._traj_cbo.release()
                if getattr(self, '_traj_vao', None) is not None:
                    try:
                        self._traj_vao.release()
                    except Exception:
                        pass
            except Exception:
                pass
            self._traj_vbo = self.ctx.buffer(segs.tobytes())
            self._traj_cbo = self.ctx.buffer(cols.tobytes())
            try:
                self._traj_vao = self.ctx.vertex_array(self.prog, [(self._traj_vbo, '3f', 'in_position'), (self._traj_cbo, '4f', 'in_color')])
            except Exception:
                self._traj_vao = None
        except Exception:
            pass

    def set_agent_trajectories(self, trajectories: np.ndarray):
        # accept shape (n_agents, history, 3)
        try:
            self._trajectories = np.asarray(trajectories, dtype='f4')
            # enforce trail_length
            if self._trajectories.shape[1] > self._trail_length:
                self._trajectories = self._trajectories[:, -self._trail_length:, :]
            self._update_traj_buffers()
            self.update()
        except Exception:
            pass

    def set_agent_directions(self, directions: np.ndarray):
        # accept shape (n_agents, 3) or (n_agents, 2); build line segments to render arrows
        try:
            dirs = np.asarray(directions, dtype='f4')
            pos = self._agent_positions
            if pos is None or dirs.shape[0] != pos.shape[0]:
                return
            segs = []
            cols = []
            for i in range(pos.shape[0]):
                a = pos[i]
                d = dirs[i]
                b = a + d
                segs.append(tuple(a.tolist()))
                segs.append(tuple(b.tolist()))
                cols.append((0.2, 0.2, 1.0, 0.9))
                cols.append((0.2, 0.2, 1.0, 0.9))
            segs = np.array(segs, dtype='f4')
            cols = np.array(cols, dtype='f4')
            # release old
            try:
                if getattr(self, '_dir_vbo', None) is not None:
                    self._dir_vbo.release()
                if getattr(self, '_dir_cbo', None) is not None:
                    self._dir_cbo.release()
                if getattr(self, '_dir_vao', None) is not None:
                    try:
                        self._dir_vao.release()
                    except Exception:
                        pass
            except Exception:
                pass
            self._dir_vbo = self.ctx.buffer(segs.tobytes())
            self._dir_cbo = self.ctx.buffer(cols.tobytes())
            try:
                self._dir_vao = self.ctx.vertex_array(self.prog, [(self._dir_vbo, '3f', 'in_position'), (self._dir_cbo, '4f', 'in_color')])
            except Exception:
                self._dir_vao = None
            self.update()
        except Exception:
            pass

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

 