"""
io.py

Preamble/Module plan for input/output adapters and format helpers (moved to fish_passage).

Responsibilities (planned):
- Read/write adapters for HEC-RAS HDF5, Flow3D, GeoJSON, and internal hdf5 cache format.
- Provide simple `read_plan(path)` and `write_simulation_state(path, sim)` APIs.
- Keep I/O functions defensive at public boundaries; internal helpers use assertions and documented exceptions.

Notes:
- Centralize HDF5 path keys and transformations to avoid duplication.
"""
