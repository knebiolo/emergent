Viewer V3

This directory contains a scaffold for a new modern OpenGL-based viewer.

Components:
- mesh_builder.py: Pure mesh builder using scipy.spatial.Delaunay.
- renderer_moderngl.py: PyQt5 QOpenGLWidget that uses moderngl for rendering.
- viewer_shim.py: Compatibility shim preserving `SalmonViewer` API.

How to run demo:

    python scripts/run_viewer_v3_demo.py

Notes:
- `moderngl` and `moderngl-window` are already present in `requirements.txt`.
- The demo is a simple smoke test; a full-featured viewer will add camera
  controls, efficient VBO updates, and simulation integration.
