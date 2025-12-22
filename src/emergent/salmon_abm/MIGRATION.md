Migration plan and status for splitting sockeye.py

Summary
-------
This document summarizes the migration that splits the monolithic `sockeye.py`
into smaller modules inside `src/emergent/salmon_abm` and tracks what's left to do.

What we implemented
--------------------
- Added `hdf5_io.py` — central HDF5 read/write helpers that accept both `h5py.File`
  and dict-like mocks used by tests.
- Extracted `movement.py`, `behavior.py`, `fatigue.py` and small `simulation.py`
  skeleton used by tests.
- Updated the monolith `sockeye.py` to emit a DeprecationWarning and to prefer
  `h5py` while supporting dict-like fallbacks for test environments.
- Converted deprecated `salmon_abm` modules to use `hdf5_io` where they
  initialize HDF outputs (notably `deprecated/sockeye_deprecated.py` and
  `deprecated/sockeye_dynamic_environment.py`).
- Added package-level re-exports in `src/emergent/salmon_abm/__init__.py` so
  callers can import `movement`, `behavior`, `fatigue`, `simulation`, `hdf5_io`.
- Parity test harness (`tests/test_parity.py`) updated and runs green.
- Tests for the `salmon_abm` package pass locally with `--basetemp=outputs/pytest_tmp`.

Files created/updated
---------------------
- src/emergent/salmon_abm/hdf5_io.py
- src/emergent/salmon_abm/movement.py (extracted)
- src/emergent/salmon_abm/behavior.py (extracted)
- src/emergent/salmon_abm/fatigue.py (extracted)
- src/emergent/salmon_abm/simulation.py (small skeleton)
- src/emergent/salmon_abm/__init__.py (re-exports)
- src/emergent/salmon_abm/sockeye.py (deprecation notice)
- src/emergent/salmon_abm/deprecated/* (selected files patched to use hdf5_io)

Remaining items from the attached plan
--------------------------------------
From the original plan in the AI journal attachment the following remain:

1) Create `utils.py` containing pure helpers (geo_to_pixel, pixel_to_geo,
   standardize_shape, determine_slices_*, calculate_front_masks, interpolation helpers).
   - Status: NOT CREATED.

2) Create `io.py` for environment import, HDF5 helper wrappers (higher-level
   helpers beyond `hdf5_io`), `movie_maker`, `output_excel`, raster/shapefile helpers.
   - Status: PARTIAL. `hdf5_io.py` exists; the higher-level IO helpers are NOT CREATED.

3) Create `pid.py` for the PID controller and add unit tests.
   - Status: NOT CREATED.

4) Expand `simulation.py` into a full simulation runner exposing the public
   `run()`, `timestep()` API used by the rest of the code.
   - Status: SMALL SKELETON exists, but not complete.

5) Create `summary.py` for reporting & summary methods.
   - Status: NOT CREATED.

6) Finalize compatibility wrapper (continue to use `sockeye.py` as shim or
   create `compat.py`) and add `__all__` lists for new modules.
   - Status: `sockeye.py` is a shim with a deprecation warning; consider adding
     a dedicated `compat.py` and sparse tests.

7) Documentation updates and migration README entry.
   - Status: PARTIAL. This `MIGRATION.md` file added; a formal `README.md`
     update and docs pages remain.

8) Run full repo tests and address non-salmon_abm regressions.
   - Status: IN-PROGRESS (next step below).

How to run tests locally (Windows)
----------------------------------
Use a repository-local pytest basetemp to avoid Windows permission issues:

```powershell
pytest -q --basetemp=outputs/pytest_tmp
```

Next recommended steps
----------------------
- Implement `utils.py`, `pid.py`, and the full `simulation.run()` API.
- Add unit tests for PID and utils.
- Optionally create `compat.py` and update `sockeye.py` to import from it.
- Run the full repository test suite and iterate until green.

If you want, I will implement `utils.py`, `pid.py`, finish `simulation.py`,
add their tests, and run the full repo test suite next.  Say "Do all" and I'll
proceed with those tasks.
