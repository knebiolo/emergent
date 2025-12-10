"""
environment.py

Preamble/Module plan for environment representation and access (moved to fish_passage).

Responsibilities (planned):
- Encapsulate raster and mesh representations; provide interpolators for depth, velocity, and other fields.
- Manage coordinate transforms and safe accessors for simulation stepping.
- Functions/classes:
  - `Environment` class with `sample_depth(x,y)`, `sample_velocity(x,y)`, and mesh access.
  - `load_hecras_plan(path)` adapter.

Notes:
- Initialization may perform larger allocations; prefer startup-only allocation pattern.
"""
