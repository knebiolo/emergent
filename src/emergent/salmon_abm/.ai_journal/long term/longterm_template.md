# Long-term Template

Metadata:
- author:
- timestamp: YYYY-MM-DD
- tags: roadmap, design, bug, research

Description:
- One-paragraph description of the long-term item and why it was elevated.

Related files:
- 

Acceptance criteria / next steps:
- 

Notes:
- 

End of long-term template.

## Long-term Goals & Vision (Salmon Module)

- Support HECRAS and Flow3D ingestion with a single `io` layer.
- Support multiple wireframe representations: heightfield, TIN, rectgrid, quadmesh, and splats.
- Provide headless render tests and offscreen reproducible smoke tests.
- Create a deterministic, headless `Simulation` API that is fully testable.
- Add CRS and reprojection utilities so agents and geometry can be aligned reliably.

## Monthly Roadmap (example)

- Month 0: Audit + architecture approval + IO adapters for Flow3D and HECRAS
- Month 1: Implement `Wireframe` interface and `HeightfieldWireframe`
- Month 2: Refactor simulation into `core/simulation.py` and add tests
- Month 3: Viewer refactor + LOD/performance + CI tests

## Checkpoints

- Daily: brief note in `reports/dailies/YYYY-MM-DD.md` with 3 bullets (done/today/blockers)
- Weekly: update `reports/weekly/YYYY-WW.md` with progress vs milestones, key risks, and decisions

## Long-term Memories (user preferences)


