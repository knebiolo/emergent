# Quick test: create a GLViewWidget and print OpenGL context/strings
import sys, time
import pytest
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Skip the test entirely if pyqtgraph.opengl isn't available in this environment
pytest.importorskip('pyqtgraph.opengl')
import pyqtgraph.opengl as gl

# Prefer desktop GL at app level
try:
    from PyQt5.QtCore import QCoreApplication
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL, True)
    print('Set AA_UseDesktopOpenGL')
except Exception as e:
    print('Failed to set AA_UseDesktopOpenGL:', e)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

w = gl.GLViewWidget()
w.setWindowTitle('GLViewWidget Test')
w.setMinimumSize(400, 300)
try:
    w.setAttribute(Qt.WA_NativeWindow, True)
    w.setAttribute(Qt.WA_PaintOnScreen, True)
    print('Set WA_NativeWindow and WA_PaintOnScreen')
except Exception as e:
    print('Setting window attributes failed:', e)

w.show()
# Print context info
try:
    from PyQt5.QtGui import QOpenGLContext
    ctx = QOpenGLContext.currentContext()
    print('QOpenGLContext.currentContext():', ctx)
    if ctx is not None:
        fmt = ctx.format()
        print('Context format:', fmt.majorVersion(), fmt.minorVersion(), fmt.profile())
    import OpenGL.GL as GL
    try:
        r = GL.glGetString(GL.GL_RENDERER)
        v = GL.glGetString(GL.GL_VENDOR)
        ver = GL.glGetString(GL.GL_VERSION)
        print('GL Vendor:', v)
        print('GL Renderer:', r)
        print('GL Version:', ver)
    except Exception as e:
        print('glGetString failed at test time:', e)
except Exception as e:
    print('Context query failed:', e)

# run briefly to allow paint events
for i in range(40):
    app.processEvents()
    time.sleep(0.05)

# try to grab an image if possible
try:
    img = w.readQImage()
    print('readQImage ok, size:', img.size())
except Exception as e:
    print('readQImage failed:', e)

w.close()
print('Test complete')
