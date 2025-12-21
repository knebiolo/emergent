"""Moderngl-based widget for rendering TIN meshes inside PyQt5.

This widget uses QOpenGLWidget and creates a ModernGL context from the
existing GL context. It expects mesh data as (verts Nx3, faces Mx3, colors Nx4).

The implementation is intentionally minimal and focuses on correctness and
visual quality. It can be extended with camera controls, lighting, and
performance optimizations (VBO reuse, frustum culling) later.
"""
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5 import QtGui
import numpy as np


def _ortho_matrix(left, right, bottom, top, near, far):
    """Return a 4x4 orthographic projection matrix (column-major).

    Matches GLSL column-major ordering for direct upload to uniform mat4.
    """
    rl = (right - left)
    tb = (top - bottom)
    fn = (far - near)
    # avoid division by zero
    if rl == 0:
        rl = 1.0
    if tb == 0:
        tb = 1.0
    if fn == 0:
        fn = 1.0
    tx = -(right + left) / rl
    ty = -(top + bottom) / tb
    tz = -(far + near) / fn
    # column-major matrix
    m = np.array([
        [2.0 / rl, 0.0, 0.0, 0.0],
        [0.0, 2.0 / tb, 0.0, 0.0],
        [0.0, 0.0, -2.0 / fn, 0.0],
        [tx, ty, tz, 1.0]
    ], dtype='f4')
    return m

try:
    import moderngl
except Exception:
    moderngl = None
try:
    import pyqtgraph as pg
    try:
        _pg_colormap = getattr(pg, 'colormap', None) or getattr(pg, 'ColorMap', None)
    except Exception:
        _pg_colormap = None
except Exception:
    _pg_colormap = None


class ModernglViewerWidget(QOpenGLWidget):
    # Emitted when an off-screen FBO preview QImage is ready (QImage object)
    fbo_preview_ready = pyqtSignal(object)
    def __init__(self, parent=None):
        super(ModernglViewerWidget, self).__init__(parent)
        # ModernGL context will be created in initializeGL
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
        # Prefer opaque painting to avoid transparent composition issues
        try:
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)
            self.setAttribute(Qt.WA_NoSystemBackground, False)
        except Exception:
            pass

    def initializeGL(self):
        if moderngl is None:
            raise RuntimeError('moderngl is not available')
        # Create moderngl context from current OpenGL context
        print('ModernglViewerWidget: initializeGL called')
        try:
            # try to create a core 3.3 context
            print('Moderngl: attempting to create context with require=330')
            self.ctx = moderngl.create_context(require=330)
        except Exception as e:
            print('Moderngl: failed to create 3.3 context:', e)
            try:
                print('Moderngl: attempting permissive context creation')
                self.ctx = moderngl.create_context()
            except Exception as e2:
                print('Moderngl: permissive context creation failed:', e2)
                raise
            else:
                print('Moderngl: context created:', type(self.ctx))
        vs = """#version 330
        in vec3 in_position;
        in vec4 in_color;
        out vec4 v_color;
        out float v_value;
        uniform mat4 mvp;
        uniform float vmin;
        uniform float vmax;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_color = in_color;
            float denom = vmax - vmin;
            if (denom == 0.0) denom = 1.0;
            v_value = (in_position.z - vmin) / denom;
        }
        """
        fs = """#version 330
        in vec4 v_color;
        in float v_value;
        out vec4 f_color;
        uniform sampler2D colormap;
        uniform int use_colormap;
        void main() {
            if (use_colormap == 1) {
                vec2 uv = vec2(clamp(v_value, 0.0, 1.0), 0.5);
                vec4 cm = texture(colormap, uv);
                f_color = cm;
            } else {
                f_color = v_color;
            }
        }
        """
        self.prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

        # create a default colormap LUT (256x1 RGBA). Prefer matplotlib if available.
        try:
            import matplotlib
            from matplotlib import cm
            cmap = cm.get_cmap('viridis')
            lut = (cmap(range(256)) * 255).astype('u1')
            # lut is Nx4 RGBA float in [0,1] -> convert to u1
            lut_bytes = lut.tobytes()
        except Exception:
            # fallback: simple blue->green->yellow->red ramp
            import numpy as _np
            lut_vals = _np.linspace(0.0, 1.0, 256)
            lut = _np.zeros((256, 4), dtype=_np.uint8)
            lut[:, 0] = (_np.clip(4.0 * (lut_vals - 0.75), 0.0, 1.0) * 255).astype(_np.uint8)
            lut[:, 1] = (_np.clip(4.0 * (lut_vals - 0.25), 0.0, 1.0) * 255).astype(_np.uint8)
            lut[:, 2] = (_np.clip(4.0 * (0.5 - lut_vals), 0.0, 1.0) * 255).astype(_np.uint8)
            lut[:, 3] = 255
            lut_bytes = lut.tobytes()

        try:
            # create 2D texture 256x1 storing RGBA u8
            self._colormap_tex = self.ctx.texture((256, 1), 4, data=lut_bytes)
            self._colormap_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._colormap_tex.repeat_x = False
            # bind sampler to texture unit 0 by default at render time
        except Exception:
            self._colormap_tex = None

        # If a mesh was provided before the GL context was ready, upload it now.
        try:
            if getattr(self, 'verts', None) is not None and getattr(self, 'faces', None) is not None and getattr(self, 'colors', None) is not None:
                try:
                    # call set_mesh to create GPU buffers now that ctx exists
                    print('Moderngl: uploading pending mesh (verts,fcs,cols)')
                    self.set_mesh(self.verts, self.faces, self.colors)
                except Exception:
                    pass
                # Try to force the window system to present the default framebuffer
                try:
                    try:
                        import OpenGL.GL as gl
                        try:
                            gl.glFlush()
                        except Exception:
                            pass
                        try:
                            gl.glFinish()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        # if Qt exposes a swapBuffers via current context, call it
                        from PyQt5.QtGui import QOpenGLContext
                        ctx = QOpenGLContext.currentContext()
                        if ctx is not None:
                            try:
                                # some Qt builds expose swapBuffers via the surface
                                surf = ctx.surface()
                                if surf is not None:
                                    try:
                                        ctx.swapBuffers(surf)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
                # Low-level GL diagnostics: query bindings and explicit glReadPixels after blit
                try:
                    import OpenGL.GL as gl
                    try:
                        fb_draw = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
                        fb_read = gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING)
                        vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
                        print('GL low-level state: DRAW_FB=', int(fb_draw), 'READ_FB=', int(fb_read), 'VIEWPORT=', tuple(vp))
                    except Exception as e:
                        print('Moderngl: failed to query GL bindings:', e)

                    # If we have a scene FBO, attempt an explicit bind+blit and then glReadPixels
                    if getattr(self, '_scene_fbo', None) is not None:
                        try:
                            src_id = getattr(self._scene_fbo, 'glo', None) or getattr(self._scene_fbo, 'framebuffer', None)
                            if src_id is not None:
                                try:
                                    src_id = int(src_id)
                                except Exception:
                                    pass
                                try:
                                    # bind read framebuffer to source and draw to 0 (default)
                                    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src_id)
                                    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
                                    gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                                    gl.glFlush()
                                except Exception as e:
                                    print('Moderngl: explicit raw glBlitFramebuffer failed:', e)
                                try:
                                    # read default framebuffer pixels directly
                                    data = gl.glReadPixels(0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
                                    import numpy as _np
                                    from PIL import Image
                                    arr = _np.frombuffer(data, dtype=_np.uint8)
                                    expected = w * h * 4
                                    if arr.size == expected:
                                        arr = arr.reshape((h, w, 4))
                                        arr = _np.flipud(arr)
                                        Image.fromarray(arr, 'RGBA').save('outputs/diag_snapshot_glread.png')
                                        print('Moderngl: saved explicit glReadPixels snapshot to outputs/diag_snapshot_glread.png')
                                    else:
                                        # handle padded rows
                                        row_bytes = arr.size // h
                                        if row_bytes >= w * 4:
                                            usable = arr[:row_bytes * h]
                                            tmp = usable.reshape((h, row_bytes))[:, :w*4].reshape((h, w, 4))
                                            tmp = _np.flipud(tmp)
                                            Image.fromarray(tmp, 'RGBA').save('outputs/diag_snapshot_glread.png')
                                            print('Moderngl: saved padded explicit glReadPixels snapshot to outputs/diag_snapshot_glread.png')
                                        else:
                                            print('Moderngl: explicit glReadPixels returned unexpected size')
                                except Exception as e:
                                    print('Moderngl: failed to glReadPixels from default framebuffer:', e)
                                finally:
                                    try:
                                        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
                                    except Exception:
                                        pass
                        except Exception as e:
                            print('Moderngl: low-level FBO->FB diagnostic failed:', e)
                except Exception:
                    pass
        except Exception:
            pass
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
        # darker neutral background to let viridis colormap stand out
        try:
            print(f'ModernglViewerWidget: paintGL viewport={self.ctx.viewport} vao_present={self._vao is not None}')
        except Exception:
            pass

        w, h = max(1, self.width()), max(1, self.height())
        # account for high-DPI scaling: framebuffer size may be widget size * devicePixelRatioF
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            try:
                dpr = float(self.devicePixelRatio())
            except Exception:
                dpr = 1.0
        fb_w = max(1, int(round(w * dpr)))
        fb_h = max(1, int(round(h * dpr)))

        # ensure a scene FBO exists with matching size
        try:
            if not hasattr(self, '_scene_fbo') or getattr(self, '_scene_fbo_size', (0, 0)) != (w, h):
                try:
                    if getattr(self, '_scene_fbo', None) is not None:
                        try:
                            self._scene_fbo.release()
                        except Exception:
                            pass
                    if getattr(self, '_scene_tex', None) is not None:
                        try:
                            self._scene_tex.release()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    self._scene_tex = self.ctx.texture((fb_w, fb_h), 4)
                    self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    self._scene_fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])
                    self._scene_fbo_size = (w, h)
                except Exception as e:
                    print('Moderngl: failed to create scene FBO:', e)
        except Exception:
            pass

        # render into the scene FBO (or default if FBO creation failed)
        try:
            if getattr(self, '_scene_fbo', None) is not None:
                self._scene_fbo.use()
            else:
                try:
                    self.ctx.screen.use()
                except Exception:
                    pass
            self.ctx.viewport = (0, 0, fb_w, fb_h)
            self.ctx.clear(0.08, 0.08, 0.12, 1.0)
            if self._vao is None:
                # nothing to draw
                try:
                    # ensure default framebuffer is bound afterwards
                    self.ctx.screen.use()
                except Exception:
                    pass
                return
            self.ctx.enable(moderngl.DEPTH_TEST)
            self._vao.render(moderngl.TRIANGLES)
        except Exception as e:
            print('Moderngl: scene render failed:', e)
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
        # --- Blit scene FBO to default framebuffer so the window compositor sees it ---
        try:
            if getattr(self, '_scene_tex', None) is not None:
                # ensure screen is bound
                try:
                    self.ctx.screen.use()
                except Exception:
                    pass
                # create blit shader/vao on demand
                try:
                    if not hasattr(self, '_blit_prog') or self._blit_prog is None:
                        vs_blit = """#version 330
                        in vec2 in_pos;
                        in vec2 in_uv;
                        out vec2 v_uv;
                        void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
                        """
                        fs_blit = """#version 330
                        in vec2 v_uv;
                        out vec4 f_color;
                        uniform sampler2D tex;
                        void main() { f_color = texture(tex, v_uv); }
                        """
                        self._blit_prog = self.ctx.program(vertex_shader=vs_blit, fragment_shader=fs_blit)
                        # fullscreen quad (pos.x,pos.y, u,v)
                        quad = np.array([
                            -1.0, -1.0, 0.0, 0.0,
                             1.0, -1.0, 1.0, 0.0,
                             1.0,  1.0, 1.0, 1.0,
                            -1.0,  1.0, 0.0, 1.0,
                        ], dtype='f4')
                        idx = np.array([0,1,2, 0,2,3], dtype='i4')
                        self._blit_vbo = self.ctx.buffer(quad.tobytes())
                        self._blit_ibo = self.ctx.buffer(idx.tobytes())
                        self._blit_vao = self.ctx.vertex_array(self._blit_prog, [(self._blit_vbo, '2f 2f', 'in_pos', 'in_uv')], index_buffer=self._blit_ibo)
                except Exception as e:
                    print('Moderngl: failed to create blit resources:', e)
                try:
                    # bind texture and draw
                    self._scene_tex.use(location=0)
                    try:
                        self._blit_prog['tex'].value = 0
                    except Exception:
                        pass
                    try:
                        self.ctx.disable(moderngl.DEPTH_TEST)
                        try:
                            # Prefer textured-quad blit (coordinates are [-1,1], texture uses normalized UVs)
                            self._blit_vao.render()
                        except Exception as e:
                            print('Moderngl: blit textured-quad failed, attempting raw glBlitFramebuffer:', e)
                            # fallback: try raw GL blit via PyOpenGL
                            try:
                                import OpenGL.GL as gl
                                # Try to get framebuffer object ids from moderngl Framebuffer
                                src_fbo = getattr(self, '_scene_fbo', None)
                                if src_fbo is not None:
                                    src_id = getattr(src_fbo, 'glo', None) or getattr(src_fbo, 'framebuffer', None)
                                    # bind read framebuffer to src and draw framebuffer to default (0)
                                    try:
                                        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, int(src_id))
                                        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
                                        gl.glBlitFramebuffer(0, 0, fb_w, fb_h, 0, 0, fb_w, fb_h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                                    finally:
                                        try:
                                            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
                                        except Exception:
                                            pass
                            except Exception as e2:
                                print('Moderngl: raw glBlitFramebuffer fallback failed:', e2)
                    except Exception as e:
                        print('Moderngl: blit render failed:', e)
                except Exception as e:
                    print('Moderngl: blit pass failed:', e)
        except Exception:
            pass
        # diagnostic framebuffer capture (balanced, robust)
        try:
            if getattr(self, '_diag_capture', False):
                # 1) Qt grab (may be transparent)
                try:
                    img = self.grabFramebuffer()
                    out = 'outputs/diag_snapshot.png'
                    img.save(out)
                    print('Moderngl: saved diagnostic framebuffer to', out)
                except Exception as e:
                    print('Moderngl: failed to save diagnostic framebuffer:', e)

                # prepare sizes (widget px and framebuffer px)
                w, h = self.width(), self.height()
                try:
                    dpr = float(self.devicePixelRatioF())
                except Exception:
                    try:
                        dpr = float(self.devicePixelRatio())
                    except Exception:
                        dpr = 1.0
                fb_w = max(1, int(round(w * dpr)))
                fb_h = max(1, int(round(h * dpr)))

                # 2) moderngl read (screen or fbo)
                if getattr(self, 'ctx', None) is not None:
                    try:
                        try:
                            self.ctx.finish()
                        except Exception:
                            try:
                                self.ctx.flush()
                            except Exception:
                                pass

                        data = None
                        try:
                            data = self.ctx.screen.read(components=4)
                        except Exception:
                            try:
                                data = self.ctx.fbo.read(components=4)
                            except Exception as e:
                                print('Moderngl: failed to read via ctx.screen/ctx.fbo:', e)

                        if data is not None:
                            try:
                                from PIL import Image
                                import numpy as _np
                                arr = _np.frombuffer(data, dtype=_np.uint8)
                                expected = fb_w * fb_h * 4
                                if arr.size == expected:
                                    arr = arr.reshape((fb_h, fb_w, 4))
                                    arr = _np.flipud(arr)
                                    # downscale to widget size if needed
                                    if (fb_w, fb_h) != (w, h):
                                        import PIL.Image as _PILImage
                                        pil = _PILImage.fromarray(arr, 'RGBA')
                                        pil = pil.resize((w, h), resample=_PILImage.NEAREST)
                                        pil.save('outputs/diag_snapshot_mgl.png')
                                    else:
                                        Image.fromarray(arr, 'RGBA').save('outputs/diag_snapshot_mgl.png')
                                    print('Moderngl: saved moderngl read snapshot to outputs/diag_snapshot_mgl.png')
                                else:
                                    print('Moderngl: moderngl read returned unexpected size', arr.size, 'expected', expected)
                            except Exception as e:
                                print('Moderngl: failed to write moderngl read snapshot:', e)

                        # 3) Off-screen FBO clear/read (solid red) as robust confirmation
                        try:
                            tex = self.ctx.texture((fb_w, fb_h), 4)
                            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                            fbo = self.ctx.framebuffer(color_attachments=[tex])
                            fbo.use()
                            self.ctx.clear(1.0, 0.0, 0.0, 1.0)
                            try:
                                self.ctx.finish()
                            except Exception:
                                try:
                                    self.ctx.flush()
                                except Exception:
                                    pass
                            try:
                                data2 = fbo.read(components=4)
                                import numpy as _np
                                from PIL import Image
                                arr2 = _np.frombuffer(data2, dtype=_np.uint8)
                                expected2 = fb_w * fb_h * 4
                                if arr2.size == expected2:
                                    arr2 = arr2.reshape((fb_h, fb_w, 4))
                                    arr2 = _np.flipud(arr2)
                                    try:
                                        # build QImage from numpy array (RGBA)
                                        h_img, w_img = arr2.shape[0], arr2.shape[1]
                                        # Qt expects bytes in native order; use frombuffer
                                        byte_data = arr2.tobytes()
                                        qimg = QtGui.QImage(byte_data, w_img, h_img, QtGui.QImage.Format_RGBA8888)
                                        # emit preview signal (downstream widget may scale)
                                        try:
                                            self.fbo_preview_ready.emit(qimg)
                                        except Exception:
                                            pass
                                    except Exception:
                                        qimg = None
                                    if (fb_w, fb_h) != (w, h):
                                        import PIL.Image as _PILImage
                                        pil = _PILImage.fromarray(arr2, 'RGBA')
                                        pil = pil.resize((w, h), resample=_PILImage.NEAREST)
                                        pil.save('outputs/diag_snapshot_fbo.png')
                                    else:
                                        Image.fromarray(arr2, 'RGBA').save('outputs/diag_snapshot_fbo.png')
                                    print('Moderngl: saved fbo snapshot to outputs/diag_snapshot_fbo.png')
                                else:
                                    print('Moderngl: fbo read unexpected size', arr2.size)
                            except Exception as e:
                                print('Moderngl: failed to read fbo:', e)
                            finally:
                                try:
                                    if fbo is not None:
                                        fbo.unuse()
                                except Exception:
                                    pass
                                try:
                                    if tex is not None:
                                        tex.release()
                                except Exception:
                                    pass
                                try:
                                    if fbo is not None:
                                        fbo.release()
                                except Exception:
                                    pass
                        except Exception as e:
                            print('Moderngl: FBO diagnostic failed:', e)
                    except Exception as e:
                        print('Moderngl: diagnostic capture flow failed:', e)

                # cleanup: clear diag flag so we only capture once
                try:
                    self._diag_capture = False
                except Exception:
                    pass
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
            print('Moderngl: ctx not ready, storing pending mesh on widget')
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
            print('Moderngl: failed to create GPU buffers for mesh')
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
        try:
            self._vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer=self.ibo)
        except Exception:
            # fallback: create VAO without color (should rarely happen)
            vao_content = [(self.vbo, '3f', 'in_position')]
            try:
                self._vao = self.ctx.vertex_array(self.prog, vao_content, index_buffer=self.ibo)
            except Exception:
                self._vao = None
        print(f'Moderngl: VAO created={self._vao is not None}, vbo={self.vbo is not None}, ibo={self.ibo is not None}, cbo={self.cbo is not None}')

        # Update MVP using mesh extents
        self._update_mvp()
        # update colormap uniforms
        try:
            vmin = float(np.min(self.verts[:, 2]))
            vmax = float(np.max(self.verts[:, 2]))
            # write uniforms
            try:
                self.prog['vmin'].value = vmin
                self.prog['vmax'].value = vmax
            except Exception:
                pass
            # bind colormap texture to unit 0 and set uniform
            if getattr(self, '_colormap_tex', None) is not None:
                try:
                    self._colormap_tex.use(location=0)
                    try:
                        self.prog['colormap'].value = 0
                        self.prog['use_colormap'].value = 1
                    except Exception:
                        pass
                except Exception:
                    try:
                        self.prog['use_colormap'].value = 0
                    except Exception:
                        pass
            else:
                try:
                    self.prog['use_colormap'].value = 0
                except Exception:
                    pass
        except Exception:
            pass
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

        # color mapping: prefer pyqtgraph colormap, otherwise simple viridis-like fallback
        try:
            vmin = float(np.nanmin(zs))
            vmax = float(np.nanmax(zs))
            denom = vmax - vmin if (vmax - vmin) != 0 else 1.0
            normed = ((zs.ravel() - vmin) / denom).clip(0.0, 1.0)
            if pg is not None and hasattr(pg, 'colormap'):
                try:
                    cmap = pg.colormap(colormap)
                    lut = cmap.getLookupTable(0.0, 1.0, 256)
                    # lut is Nx3 or Nx4; interpolate
                    idx = (normed * (lut.shape[0] - 1)).astype(int)
                    rgba = lut[idx]
                    if rgba.shape[1] == 3:
                        alphas = np.ones((rgba.shape[0], 1), dtype=rgba.dtype)
                        rgba = np.concatenate([rgba, alphas], axis=1)
                    colors = np.asarray(rgba, dtype='f4')
                except Exception:
                    colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype='f4'), (verts.shape[0], 1))
            else:
                # simple viridis-like fallback using a small hardcoded palette
                try:
                    # create a simple gradient from blue->green->yellow
                    def _simple_viridis(v):
                        # v in [0,1]
                        r = np.clip(4.0 * (v - 0.75), 0.0, 1.0) + np.clip(4.0 * (v - 0.5), 0.0, 1.0) * 0.0
                        g = np.clip(4.0 * (v - 0.25), 0.0, 1.0)
                        b = np.clip(4.0 * (0.5 - v), 0.0, 1.0)
                        return np.stack([r, g, b, np.ones_like(r)], axis=1)
                    colors = _simple_viridis(normed)
                    colors = np.asarray(colors, dtype='f4')
                except Exception:
                    colors = np.tile(np.array([0.7, 0.7, 0.7, 1.0], dtype='f4'), (verts.shape[0], 1))
        except Exception:
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

 