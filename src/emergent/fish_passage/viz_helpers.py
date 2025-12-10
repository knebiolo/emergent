"""
viz_helpers.py

Preamble/Module plan for converting simulation state into renderable payloads (moved to fish_passage).

Responsibilities (planned):
- Build lightweight payloads `{verts, faces, colors}` for mesh rendering and scatter plots for agent points.
- Provide helpers to map scalar metrics to color ramps and simplify mesh geometry for performance.

Notes:
- Keep transformations pure and unit-testable; avoid side-effect-heavy conversions inside viewer event loop.
"""
