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

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class HECRASMap:
  """Lightweight container caching coordinates and field arrays for HECRAS mapping.

  This is a minimal stub to allow imports and will be filled in during the full port.
  """
  coords: Optional[np.ndarray] = None
  fields: Dict[str, np.ndarray] = None

  def map_point(self, x: float, y: float) -> Dict[str, Any]:
    """Return a dictionary of mapped field values for a point (x,y).

    Stub implementation: returns empty dict if no fields available.
    """
    if not self.fields:
      return {}
    # naive nearest neighbor using simple euclidean distance over coords
    if self.coords is None or len(self.coords) == 0:
      return {}
    dists = np.sum((self.coords - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(dists))
    return {k: v[idx] for k, v in self.fields.items()}

