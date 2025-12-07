"""Quick diagnostics: check PyQt5, pyqtgraph.opengl, and OpenGL context availability without opening a visible window."""
import sys
import traceback

def run():
    print('Python executable:', sys.executable)
    try:
        import PyQt5
        print('PyQt5 imported:', PyQt5)
    except Exception as e:
        print('PyQt5 import failed:', e)
        traceback.print_exc()
        return
    try:
        from PyQt5 import QtWidgets, QtGui, QtCore
        print('QApplication available')
    except Exception as e:
        print('QApplication import failed:', e)
        traceback.print_exc()
        return
    try:
        import pyqtgraph as pg
        print('pyqtgraph imported, version:', getattr(pg, '__version__', 'unknown'))
    except Exception as e:
        print('pyqtgraph import failed:', e)
    try:
        import pyqtgraph.opengl as gl
        print('pyqtgraph.opengl imported:', gl)
    except Exception as e:
        print('pyqtgraph.opengl import failed:', e)
    try:
        # Try creating a QOpenGLContext (no visible widget)
        from PyQt5.QtGui import QSurfaceFormat, QOpenGLContext
        fmt = QSurfaceFormat()
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setVersion(3, 3)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        ok = ctx.create()
        print('QOpenGLContext create() =>', ok)
    except Exception as e:
        print('QOpenGLContext creation failed:', e)
        traceback.print_exc()

if __name__ == '__main__':
    run()
