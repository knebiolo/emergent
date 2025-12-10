"""
hecras.py

Preamble/Module plan for centralized HECRAS helpers.

Responsibilities (planned):
- Centralize all HECRAS HDF5 parsing, coordinate transforms, raster conversion, and perimeter extraction.
- Provide a single `HECRASMap` class that caches KDTree and field arrays for fast IDW mapping.
- Expose functions:
  - `infer_wetted_perimeter(hdf_path_or_file, ...)`
  - `compute_affine_from_hecras(coords, ...)`
  - `map_fields_to_raster(...)`
- Ensure interfaces are pure (no hidden global state) and safe to call from worker threads.

Notes:
- Existing helpers in `hecras_helpers.py` will be refactored into this module.
- Avoid dynamic allocations at runtime where possible; allocate buffers during initialization.
"""
