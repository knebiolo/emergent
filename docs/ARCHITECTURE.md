**Emergent Repo — High-Level Architecture**

- **Purpose**: single-file reference describing module responsibilities, data ownership, and important runtime defaults to avoid duplicated logic and drift.

Core modules (one-liners):
- `src/emergent/salmon_abm/sockeye.py` (formerly `sockeye_SoA_OpenGL_RL.py`): Simulation core — owns HECRAS ingestion, wetted-perimeter inference, `perimeter_points` / `perimeter_polygon` / `wetted_mask`, agent state, and PID controllers. This is the authoritative source for environment geometry.

---
**Proposed Ground-up Salmon Module Architecture (2025-12-09)**

Overview
--------
The salmon ABM and visualization stack will be refactored into a modular, testable architecture that cleanly separates IO, geometry processing, simulation core, and visualization. The redesign will add support for Flow3D inputs and multiple wireframe representations (not just TINs), support 2D (depth-averaged) and 3D datasets natively, and provide headless rendering and testability.

Key Principles
--------------
- Single authoritative IO layer for HECRAS, Flow3D, and other formats
- Wireframe abstraction to allow multiple geometry backends (TIN, heightfield, rectgrid, quadmesh, splats)
- Simulation core must be headless-first and deterministic for tests
- Viewer is a pluggable visualization that accepts wireframes and simulation attachments
- Clear data models (dataclasses) for plan, wireframe payloads, and agent state

Top-level Components
--------------------
- io: readers/writers for HECRAS HDF5, Flow3D outputs, TIN payloads, and raster exports
- geom: meshing, alpha-shapes, sampling, transforms, and wireframe implementations
- hecras_helpers: high-level HECRAS logic (perimeter inference, mapping, centerline)
- core: simulation engine, environment, agents, step loop, and metrics
- view: GUI and offscreen rendering, mesh worker threads, and rendering pipeline
- util: logging, safe exception helpers, KDTree wrappers, reprojection utilities

Wireframe Abstraction
---------------------
A small interface that all geometry backends implement; viewer and sim accept a Wireframe instance. Implementors include:
- `TINWireframe`: Delaunay/TIN from irregular points
- `HeightfieldWireframe`: Raster/DEM / gridded heightfield (recommended default for dense outputs)
- `RectGridWireframe`: Structured CFD/Flow3D grids
- `QuadMeshWireframe`: Regular quad meshes for efficient rendering
- `SplatCloudWireframe`: Point splats for huge point clouds

Flow3D Support
---------------
Add `io/flow3d.py` as a thin adapter for Flow3D outputs (HDF/VTK/NetCDF). Convert Flow3D outputs to `HeightfieldWireframe` by default for performance, with optional TIN conversion for precision needs.

2D/3D Support
-------------
Wireframes and IO will support both 2D depth-averaged datasets and full 3D outputs. Data models make dimensionality explicit; viewers and simulations apply transforms accordingly.

Migration Strategy
------------------
Phased approach: Audit -> IO adapters -> wireframe abstraction -> simulation refactor -> viewer refactor -> CI and tests. See `reports/architecture_proposal_2025-12-09.md` for detailed roadmap and milestones.

---
- `src/emergent/salmon_abm/salmon_viewer_v2.py`: GL viewer — visualizes the simulation; reads perimeter and mesh payloads from `sim` (does NOT compute perimeter). Performs mesh clipping for display only.
- `src/emergent/salmon_abm/tin_helpers.py`: Spatial helpers — `sample_evenly()` for stratified sampling and `alpha_shape()` for concave hull polygonization.
- `tools/`: Experiment and diagnostic scripts — exploratory code, not authoritative. These may be archived or removed after integration.
- `data/`: External data (HECRAS .hdf files, start polygon). Use `data/salmon_abm/...` for HECRAS inputs.
- `outputs/`: Generated artifacts and previews. Large experimental outputs should be pruned or moved to `outputs/archive/` after verification.

Key invariants (single-source rules):
- Wetted-perimeter MUST be computed by the simulation at startup and stored on the sim object (`sim.perimeter_points`, `sim.perimeter_polygon`, `sim.wetted_mask`). Viewers and ABM code MUST read these attributes and treat the sim as authoritative.
- Viewer MUST NOT compute or re-infer the perimeter in production. Viewer may have lightweight preview helpers only for developer convenience (but prefer disabled by default). Recent changes removed viewer fallback perimeter logic.
- Default HECRAS perimeter settings:
  - `hecras_perim_depth` default: `1e-5` (meters)
  - `hecras_perim_timestep` default: middle timestep when available
  - `tin_max_nodes` default: 5000 (viewer sampling limit)
- Polygonization: prefer vector-first `alpha_shape()` (concave hull). If that fails, fallback to convex hull. Store result as a Shapely geometry on `sim.perimeter_polygon`.

Developer workflow notes:
- When changing perimeter inference, update the canonical simulation module (`src/emergent/salmon_abm/sockeye.py`) and ensure `sim.perimeter_*` attributes are set. Legacy filenames (e.g. `sockeye_SoA_OpenGL_RL.py`) may still appear in older logs; treat `sockeye.py` as the source of truth.
- Avoid copying inference logic into viewers, tools, or ABM modules. If a tool needs to run experiments, keep it under `tools/` and mark it as experimental.
- Keep `tin_helpers.py` small and dependency-light (alpha-shape uses `scipy` + `shapely`).

Recent edits (integration summary):
- Simulation now computes HECRAS wetted perimeter on init using `infer_wetted_perimeter_from_hecras(...)` and polygonizes using `tin_helpers.alpha_shape()`; outputs are attached to `sim`.
- Viewer `salmon_viewer_v2.py` now reads `sim.perimeter_points` and `sim.perimeter_polygon` and clips TIN triangles for display; viewer fallback logic removed.

How to update this document:
- Edit `docs/ARCHITECTURE.md` for any change in ownership or data flow.
- Add a short changelog entry and increment the module index if modules are refactored.

Quick contacts for repo knowledge:
- If something looks duplicated, first check `ARCHITECTURE.md` before editing.

