"""Simple launcher to start SalmonViewer (v3 shim) and load the latest outputs mesh.

Usage: python scripts/launch_viewer.py

This creates a minimal `sim` placeholder with no heavy dependencies and uses
`SalmonViewer.load_hecras_mesh` to load a prebuilt mesh `.npz` from `outputs/`.
"""
import sys, os, glob
import numpy as np
from PyQt5 import QtWidgets
from emergent.salmon_abm.viewer_v3.viewer_shim import SalmonViewer


def find_latest_mesh():
    outdir = os.path.join(os.getcwd(), 'outputs')
    if not os.path.isdir(outdir):
        return None
    files = glob.glob(os.path.join(outdir, '*_mesh.npz'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


class DummySim:
    def __init__(self):
        self.hdf5 = None
        self.perimeter_points = None
        self.vert_exag = 1.0


def print_gl_info():
    try:
        from PyQt5.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat.defaultFormat()
        print('QSurfaceFormat: version', fmt.majorVersion(), fmt.minorVersion(), 'profile', fmt.profile())
    except Exception:
        print('QSurfaceFormat: unavailable')
    # Note: this launcher prefers the CPU renderer; do not probe ModernGL here.
    print('Renderer preference: CPU renderer (ModernGL not used)')


def main(argv=None):
    argv = argv or sys.argv[1:]
    diag = '--diag' in argv or '--diagnostic' in argv
    # Allow diag variants via env or explicit flag
    variant = None
    for a in argv:
        if a.startswith('--diag-variant='):
            try:
                variant = int(a.split('=')[1])
            except Exception:
                variant = None
    try:
        from PyQt5.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        # variant overrides
        no_alpha = os.environ.get('EMERGENT_QSURFACE_NO_ALPHA') == '1'
        compat = os.environ.get('EMERGENT_QSURFACE_COMPAT') == '1'
        if variant is not None:
            if variant == 1:
                no_alpha = True
            elif variant == 2:
                compat = True
        if compat:
            try:
                # try requesting compatibility profile if available
                fmt.setProfile(QSurfaceFormat.NoProfile)
            except Exception:
                fmt.setProfile(QSurfaceFormat.CoreProfile)
        else:
            fmt.setProfile(QSurfaceFormat.CoreProfile)
        if no_alpha:
            fmt.setAlphaBufferSize(0)
        QSurfaceFormat.setDefaultFormat(fmt)
        print('Requested QSurfaceFormat 3.3', 'compat' if compat else 'core', 'no_alpha' if no_alpha else '')
    except Exception:
        print('Failed to set QSurfaceFormat; continuing with default')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sim = DummySim()
    viewer = SalmonViewer(sim)
    # force CPU preview unconditionally to avoid any GL widget
    try:
        viewer._force_cpu = True
    except Exception:
        pass
    # Also set diag mode if requested (keeps earlier behavior)
    if diag:
        try:
            viewer._diag_mode = True
        except Exception:
            pass
    # try to load the latest mesh file directly
    mesh = find_latest_mesh()
    if mesh is not None:
        try:
            d = np.load(mesh)
            verts = d['verts']
            faces = d['faces']
            colors = d['colors']
            # prefer using loader API to allow pending upload if ctx not ready
            try:
                viewer.load_tin_payload({'verts': verts, 'faces': faces, 'colors': colors})
            except Exception:
                try:
                    if viewer.gl_widget is not None:
                        viewer.gl_widget.set_mesh(verts, faces, colors, vert_exag=getattr(viewer.sim, 'vert_exag', 1.0))
                    else:
                        viewer.last_mesh_payload = {'verts': verts, 'faces': faces, 'colors': colors}
                except Exception:
                    pass
            print('Loaded mesh', mesh)
            if diag:
                print_gl_info()
                try:
                    gw = viewer.gl_widget
                    print('gl_widget:', 'exists' if gw is not None else 'None')
                    if gw is not None:
                        print('  ctx:', type(getattr(gw, 'ctx', None)), 'vao:', getattr(gw, '_vao', None) is not None)
                except Exception as e:
                    print('  error inspecting gl_widget:', e)
        except Exception:
            print('Failed to load mesh file:', mesh)
    else:
        print('No mesh file found in outputs/')
    try:
        # build full UI (this calls QApplication.exec_ internally)
        if diag:
            print('Running viewer in diagnostic mode...')
        # If diagnostic mode, schedule a delayed inspection after the event loop starts
        if diag and viewer.gl_widget is not None:
            try:
                from PyQt5.QtCore import QTimer
                def inspect_gl():
                    try:
                        from PyQt5.QtGui import QOpenGLContext
                        ctx = QOpenGLContext.currentContext()
                        if ctx is not None:
                            fmt = ctx.format()
                            print('Current QOpenGLContext format: version', fmt.majorVersion(), fmt.minorVersion(), 'profile', fmt.profile())
                        else:
                            print('No current QOpenGLContext')
                    except Exception as e:
                        print('Error inspecting QOpenGLContext:', e)
                    try:
                        gw = viewer.gl_widget
                        print('gl_widget inspection: exists' if gw is not None else 'gl_widget None')
                        if gw is not None:
                            try:
                                cls = gw.__class__
                                print('  widget class:', cls.__name__, 'module:', cls.__module__)
                                uses_cpu = ('cpu' in cls.__module__.lower()) or ('FastCPU' in cls.__name__) or ('CPUViewerWidget' in cls.__name__)
                                print('  using_cpu_renderer:', uses_cpu)
                                try:
                                    print('  widget size:', gw.width(), gw.height())
                                except Exception:
                                    pass
                            except Exception as e:
                                print('  error reading gl_widget attributes:', e)
                    except Exception as e:
                        print('Error inspecting gl_widget:', e)
                QTimer.singleShot(500, inspect_gl)
            except Exception:
                pass
        # enable framebuffer capture on the widget for diagnostic snapshot
        try:
            if diag and getattr(viewer, 'gl_widget', None) is not None:
                try:
                    viewer.gl_widget._diag_capture = True
                except Exception:
                    pass
            # show the FBO preview overlay if available (helps when on-screen GL composition fails)
            try:
                if diag and hasattr(viewer, 'show_fbo_preview'):
                    try:
                        from PyQt5.QtCore import QTimer
                        # show preview shortly after startup so the diag capture has time to write the file
                        def show_preview_later():
                            try:
                                viewer.show_fbo_preview(True)
                            except Exception:
                                pass
                        QTimer.singleShot(800, show_preview_later)
                    except Exception:
                        try:
                            viewer.show_fbo_preview(True)
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass
        # Load the latest preview image into the UI immediately (CPU-first behavior)
        try:
            from PyQt5.QtCore import QTimer
            from PyQt5.QtGui import QPixmap

            def load_preview_into_ui():
                try:
                    p = os.path.join(os.getcwd(), 'outputs', 'latest_preview.png')
                    if os.path.exists(p) and getattr(viewer, '_fbo_preview_label', None) is not None:
                        pix = QPixmap(p)
                        lbl = viewer._fbo_preview_label
                        try:
                            pix = pix.scaled(lbl.width(), lbl.height())
                        except Exception:
                            pass
                        lbl.setPixmap(pix)
                        lbl.setVisible(True)
                except Exception:
                    pass

            QTimer.singleShot(200, load_preview_into_ui)
        except Exception:
            pass
        viewer.run()
        return
    except Exception as e:
        print('Failed to start Qt/OpenGL viewer:', e)
        # fallback: open the most recent preview PNG if available
        outdir = os.path.join(os.getcwd(), 'outputs')
        candidates = []
        if os.path.isdir(outdir):
            candidates = glob.glob(os.path.join(outdir, '*_preview*.png'))
        if candidates:
            latest_preview = max(candidates, key=os.path.getmtime)
            print('Opening preview image instead:', latest_preview)
            try:
                # Windows: launch default image viewer
                os.startfile(latest_preview)
            except Exception:
                try:
                    # fallback: show via matplotlib
                    import matplotlib.pyplot as plt
                    img = plt.imread(latest_preview)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                except Exception as e2:
                    print('Failed to open preview image:', e2)
        else:
            print('No preview image found in outputs/')


if __name__ == '__main__':
    main()
