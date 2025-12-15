"""
hecras.py

Preamble/Module plan for centralized HECRAS helpers (moved to fish_passage).

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

"""
This module forwards HECRAS-specific responsibilities to the centralized
implementation in `emergent.fish_passage.io` to avoid duplicated logic.

The original `HECRASMap` lived in multiple places; prefer the tested
implementation in `io.HECRASMap` and adapter helpers like
`map_hecras_for_agents`.
"""

from typing import Any
from emergent.fish_passage.io import HECRASMap as IO_HECRASMap, map_hecras_for_agents

# Thin wrapper for compatibility with older imports. Use IO_HECRASMap directly
# in new code; this module remains to avoid wide import churn.

def HECRASMap(*args, **kwargs) -> IO_HECRASMap:
    """Return an instance of the canonical HECRASMap implementation.

    This wrapper preserves the simple constructor signature used by callers
    while ensuring only one canonical implementation exists.
    """
    return IO_HECRASMap(*args, **kwargs)


__all__ = ['HECRASMap', 'map_hecras_for_agents']

