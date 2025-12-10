"""
rendering.py

Preamble/Module plan for rendering utilities used by the Salmon ABM (moved to fish_passage).

Responsibilities (planned):
- GL mesh building workers and safe fallbacks to CPU-based rasterization.
- Classes:
  - `GLMeshBuilder` (background thread) — triangulates input points, computes vertex buffers.
  - `OffscreenRenderer` — CPU painter-based renderer using Qt QImage/QPainter as fallback.
  - `GLRenderer` — thin wrapper around GL view when available.
- Export simple payload format: dict with 'verts', 'faces', 'colors' for viewers to consume.
- Keep functions small and testable; avoid global mutable state.

Notes:
- Ensure any loops in heavy routines have clear, provable bounds where possible (aspirational).
- Startup-only allocation for large buffers; workers receive buffers by reference where appropriate.
"""
