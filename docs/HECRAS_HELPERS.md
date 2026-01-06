HECRAS Helpers

Overview
- Design note for centralized utilities for working with HECRAS HDF5 plan files.
- Status: the Salmon ABM production run path does not currently ingest HECRAS directly; these helpers are planned and some legacy prototypes may exist under `src/emergent/salmon_abm/deprecated/`.

Timestep semantics
- Many HECRAS datasets are timeseries with shape (T, N). The helper functions accept a `timestep` parameter (default 0).
- If `timestep` is larger than available indices, the last available timeslice is used.

Suggested usage
- For agent mapping: use `map_hecras_to_env_rasters(simulation, plan_path, field_names=[...], k=1)` which will forward `timestep` if present on `simulation`.
- For extraction: implement `infer_wetted_perimeter_from_hecras(...)` first (see `docs/HECRAS_HDF5_STRUCTURE.md`) before relying on wetted-perimeter-derived geometry.

Notes
- The helpers will try a vector extraction first (using HECRAS facepoints/perimeter datasets) and fall back to a coarse raster approach when needed.
- Prefer running tests in CI (Linux) where binary wheels are consistent; on Windows, prefer `conda-forge` installs for `h5py` and `PyOpenGL`.
