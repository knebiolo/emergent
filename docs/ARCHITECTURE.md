# Emergent Repo — High-Level Architecture

Purpose: single-file reference describing module responsibilities, data ownership, and important runtime defaults to avoid duplicated logic and drift.

## Core Modules (Salmon ABM)

- `src/emergent/salmon_abm/simulation.py`: Current simulation core (raster-driven). Owns agent state, stepping (`timestep`/`run`), and output writing.
- `src/emergent/salmon_abm/io.py` + `src/emergent/salmon_abm/hdf5_io.py`: Environment ingestion + persistence. Writes/reads static rasters (depth/velocity) into the HDF5 DB under `environment/*`.
- `src/emergent/salmon_abm/realtime_viewer.py`: Runtime viewer. Visualizes simulation outputs from the HDF5 DB.
- `src/emergent/salmon_abm/rl_training.py` + `src/emergent/salmon_abm/rl_training_viewer.py`: RL training + visualization. Optimizes `BehavioralWeights` and applies them via `sim.load_behavioral_weights(...)`.
- `src/emergent/salmon_abm/sockeye.py`: Deprecated legacy/compatibility shim (kept for parity tests and historical reference). Do not add new imports/features here; port forward into `simulation.py` and related modules.
- `tools/`: Experiment and diagnostic scripts. Prefer editing existing scripts (e.g., `tools/test_salmon_abm.py`, `tools/run_salmon_production.py`) rather than adding new ones.
- `data/`: External data. Current runs use `data/salmon_abm/*.tif` (static rasters) plus start polygons; direct HECRAS ingestion is not wired into the production run path yet.
- `outputs/`: Generated artifacts and previews.

## Key Invariants

- The Salmon ABM run path is raster-based by default, but `simulation.py` now supports an optional HECRAS direct mode that maps time-varying HECRAS fields directly to agent positions (with static t0 rasters used for distance-to-bank).
- Do not assume any direct-HECRAS mesh objects (e.g., `sim.perimeter_*`) exist in production; only per-agent HECRAS sampling is supported in the new mode.
- When the direct HECRAS interface is implemented, the simulation should remain the single source of truth for derived geometry (e.g., wetted masks/perimeters); viewers should only visualize.

## Developer Workflow Notes

- When migrating code out of `sockeye.py`, port into focused modules under `src/emergent/salmon_abm/` and update callers to import the new locations.
- Avoid copying core logic into viewers or tools. If a tool needs experimental logic, keep it under `tools/` and clearly label it as experimental.

## Updating This Document

- Edit `docs/ARCHITECTURE.md` for any change in ownership or data flow.
