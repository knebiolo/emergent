fish_passage
============

Minimal scaffold demonstrating coding policies:

- Small, focused functions
- Bounded loops with explicit assertions
- Startup-only allocation example
- Minimal defensive handling and clear assertions

Functions:

- `compute_travel_times(distances, speed)` — computes travel times with input validation.
- `summarize_times(times)` — returns (min, max) for a times list.

HECRAS Integration
------------------

The canonical API for obtaining per-agent flow components is
`get_agent_flow_components(sim, k=None)` in `src/emergent/fish_passage/io.py`.
This helper follows the HECRAS-first policy:

- Prefer HECRAS nodal mapping when `sim.use_hecras` is enabled and a
	`sim.hecras_plan_path` is configured.
- Fall back to `environment/vel_x` and `environment/vel_y` rasters in `sim.hdf5`.
- Fall back to `environment/vel_dir` raster and convert to unit vectors.

For backward compatibility, `src/emergent/fish_passage/hecras.py` provides a
thin wrapper to return the canonical `HECRASMap` from `io.py` so callers can
migrate in small, test-covered steps.
