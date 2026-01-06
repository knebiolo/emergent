**HECRAS Integration (Planned / Not Yet Wired)**

Status note: as of early 2026 the Salmon ABM production run path still imports **static environmental rasters** (e.g., `data/salmon_abm/depth.tif`, `vel_x.tif`, `vel_y.tif`). Direct ingestion of HECRAS plan HDF5 files is not currently wired into `src/emergent/salmon_abm/simulation.py`.

This document is a design/reference note for a future "HECRAS-only mode" where HECRAS HDF5 plan files are used as a read-only source for environmental fields (`depth`, `x_vel`, `y_vel`, `wsel`, etc.) via an IDW k-NN mapping (fast, approximate).

Planned interface (not implemented yet):

- `hecras_plan_path` (string): path to the HECRAS HDF5 plan file.
- `hecras_fields` (list): list of field names to map from the HECRAS HDF (e.g. `['Cells Minimum Elevation','Water Surface','Cell Velocity - Velocity X','Cell Velocity - Velocity Y']`).
- `hecras_k` (int): number of nearest neighbors for IDW mapping (default: 8).
- `use_hecras` (bool): if `True`, the HECRAS-mapped fields will override raster-derived `depth`/`x_vel`/`y_vel` attributes.

Headless benchmark runner

- There is currently no canonical `tools/run_headless_hecras_sim.py` runner in the repo.
- When HECRAS-only mode is implemented, add a dedicated benchmark runner under `tools/` and keep it explicitly marked as experimental until validated.

Notes
- When running with `use_hecras=True` the code will avoid raster imports and derive safe transforms
  from the HECRAS coordinates if needed. The HECRAS HDF files are treated as read-only inputs.
