OpenGL setup and verification

This document explains how to install and verify OpenGL/PyOpenGL support for the Emergent project.

Conda (recommended)

1. Activate the project's environment:

```powershell
conda activate emergent
```

2. Install PyOpenGL and the accelerator from conda-forge:

```powershell
conda install -c conda-forge pyopengl pyopengl_accelerate -y
```

3. (Optional) If you prefer pip for these packages:

```powershell
pip install PyOpenGL PyOpenGL_accelerate
```

Verification

Run the following short Python check to confirm `pyqtgraph.opengl` can import:

```powershell
python -c "import importlib; importlib.import_module('pyqtgraph.opengl'); print('pyqtgraph.opengl: AVAILABLE')"
```

Troubleshooting

- If you see `ModuleNotFoundError: No module named 'OpenGL'`, ensure `PyOpenGL` is installed in the same interpreter/environment you're using to run the project.
- On Windows, GPU drivers and system OpenGL support matter. Ensure your GPU drivers are up to date.
- If `pyqtgraph.opengl` imports but the GL window doesn't appear or is blank, try running a minimal GL example and check for `QOpenGLContext` initialization errors.

Quick test script

Save the following as `tools/test_gl.py` and run it. It opens a simple GLViewWidget and should display an empty 3D view.

```python
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
view = gl.GLViewWidget()
view.show()
app.exec_()
```

If this runs without errors and an empty GL window appears, OpenGL integration is available.
