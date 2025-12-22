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
