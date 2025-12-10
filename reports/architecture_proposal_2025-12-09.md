# Architecture Proposal — Salmon Module

Date: 2025-12-09

Summary
-------
This document proposes a ground-up modular architecture for the `salmon` module. It incorporates Flow3D and HECRAS IO, adds a wireframe abstraction supporting multiple mesh/backends (TIN, heightfield, rectgrid, quadmesh, splats), and documents a phased migration roadmap to move existing code into cleanly separated packages.

Decisions
---------
- Use a single canonical `io` layer for all external formats.
- Introduce a `Wireframe` abstraction for geometry representation; viewer and simulation will consume this API.
- Default rendering path for dense CFD/Flow3D outputs is `HeightfieldWireframe` for performance.
- Maintain `hecras_helpers.py` as the canonical HECRAS algorithm collection; expose IO wrappers in `io/hecras.py` that delegate to the helpers.

Detailed Components
-------------------
(See `docs/ARCHITECTURE.md` for the top-level description)

Wireframe Types & Tradeoffs
---------------------------
- Heightfield: fastest for dense grids; recommended default for Flow3D.
- TIN: precise for irregular points; expensive for large N.
- RectGrid: preserves CFD native structure.
- QuadMesh: good LOD and rendering performance.
- Splats: point-based rendering for extremely large point clouds.

2D/3D Considerations
--------------------
All IO and Wireframe types will support dimension metadata. The system will be able to ingest 2D depth-averaged HECRAS results and 3D Flow3D datasets; the `Wireframe` will expose dimensionality so simulation and viewer can adapt.

Migration Roadmap (high level)
------------------------------
- Phase 0: Audit current usage and tests (2–3 days)
- Phase 1: Implement IO adapters for Flow3D and consolidate HECRAS IO (1 week)
- Phase 2: Implement `Wireframe` base + `HeightfieldWireframe` and `TINWireframe` (2 weeks)
- Phase 3: Refactor `sockeye.py` into `core/simulation.py` exposing the Simulation API (2 weeks)
- Phase 4: Update viewer to accept `Wireframe` and add offscreen test harness (2–4 weeks)
- Phase 5: CI, performance tuning, documentation (ongoing)

Acceptance Criteria
-------------------
- Flow3D file can be ingested and rendered as a heightfield with agents overlaid correctly.
- Tests verifying headless import and agent-overlay run in local CI (or documented manual steps if CI cannot run GUI components).

Next steps
----------
1. Approve the architecture and roadmap.
2. I will produce an audit mapping existing files to the new component layout.
3. Create the `io/flow3d.py` skeleton and `geom/wireframe.py` base and `heightfield` implementation.
