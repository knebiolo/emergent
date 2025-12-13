import faulthandler
import sys
import traceback
modules = [
    'numpy', 'scipy', 'h5py', 'pyqt5', 'PyQt5', 'pyqtgraph', 'pyqtgraph.opengl', 'OpenGL', 'OpenGL.GL', 'matplotlib', 'shapely']
print('Starting import diagnostics')
for m in modules:
    try:
        print('Importing', m)
        __import__(m)
        print(' OK', m)
    except Exception as e:
        print(' FAIL import', m, type(e), e)
        traceback.print_exc()
    except BaseException as e:
        print(' CRASH on import', m, repr(e))
        raise
print('Diagnostics complete')
