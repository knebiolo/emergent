**HECRAS Integration (HECRAS-only Mode)**

This project supports using HECRAS HDF5 plan files as a read-only source for environmental fields
(`depth`, `x_vel`, `y_vel`, `wsel`, etc.) via an IDW k-NN mapping (fast, approximate).

Enable HECRAS-only mode by providing the following arguments to `simulation` (or via the headless runner):

- `hecras_plan_path` (string): path to the HECRAS HDF5 plan file.
- `hecras_fields` (list): list of field names to map from the HECRAS HDF (e.g. `['Cells Minimum Elevation','Water Surface','Cell Velocity - Velocity X','Cell Velocity - Velocity Y']`).
- `hecras_k` (int): number of nearest neighbors for IDW mapping (default: 8).
- `use_hecras` (bool): if `True`, the HECRAS-mapped fields will override raster-derived `depth`/`x_vel`/`y_vel` attributes.

Headless benchmark runner

There is a helper script `tools/run_headless_hecras_sim.py` that runs the simulation in HECRAS-only
mode and records per-timestep `environment()` timings to CSV.

Usage (PowerShell):

```powershell
python tools/run_headless_hecras_sim.py --timesteps 200 --num-agents 200 --out outputs/hecras_benchmark.csv
```

The CSV contains columns: `timestep`, `duration_s`, `sample_depth_0`, `sample_xvel_0`.

Notes
- When running with `use_hecras=True` the code will avoid raster imports and derive safe transforms
  from the HECRAS coordinates if needed. The HECRAS HDF files are treated as read-only inputs.

Canonical API (fish_passage)
---------------------------

During the migration the canonical programmatic API for consumers that need per-agent
flow components is `get_agent_flow_components(sim, k=None)` provided by
`src/emergent/fish_passage/io.py`. This helper implements the project policy of
"HECRAS-first, raster fallback":

- If `sim.use_hecras` is True and a `sim.hecras_plan_path` is set, the function maps
  the HECRAS nodal fields (e.g. `Velocity X`, `Velocity Y`) to the agent XY positions
  using the KDTree IDW adapter.
- Else, it attempts to sample raster datasets `environment/vel_x` and `environment/vel_y`
  from `sim.hdf5` using existing raster transforms.
- Else, if only `environment/vel_dir` is present, it converts the direction raster into
  unit vector components via cosine/sine.
- If none of the above are available, it returns NaN arrays so callers can detect
  missing flow.

Consumers should call `get_agent_flow_components(sim, k=...)` rather than reading
rasters directly; this centralizes semantics and makes migrating legacy callers
straightforward. For backward compatibility, `src/emergent/fish_passage/hecras.py`
exposes a thin wrapper that returns the canonical `HECRASMap` implementation from
`io.py`.
