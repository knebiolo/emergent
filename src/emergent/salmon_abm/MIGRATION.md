Migration plan and status for splitting sockeye.py

Summary
-------
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

Outstanding Migration Items (2025-12-22)
-------------------------------------

- Goal: finish the migration so the new `simulation` runner compiles and can execute timesteps and HDF5 writes. Postpone summary and Kaplan-Meier work.

- Findings from scan of `sockeye.py`:
  - Most domain logic (movement, behavior, fatigue, PID, I/O) is already present in the new modules. `sockeye.py` remains a compatibility shim.
  - `initialize_hdf5()` now delegates to `io.write_sim_initial()` and has robust fallbacks that use `hdf5_io`.
  - Many functions still use direct `self.hdf5[...]` access; tests and the new `io` layer rely on `hdf5_io.get_hdf5_obj(...)` and `hdf5_io.read_dataset/write_dataset` to support dict-backed tests. A few direct reads can fail when `self.hdf5` is a dict; these should be guarded or replaced.
  - `kaplan_meier_estimator` usage in `summary.kaplan_curve()` depends on `sksurv` — this is deferred as requested (backburner).
  - Several TODO items are domain tuning (ucrit, PID scaling, buffer sizes). These are design/QA tasks and do not block compilation.

- Immediate migration tasks I will (or have) prioritize to get simulations runnable:
  1. Make simple, safe IO fallbacks in `sockeye.py` so any reads use `hdf5_io.get_hdf5_obj(self)` or `hdf5_io.read_dataset` whenever reasonable (not exhaustive; focus on reads used during `timestep()` and `run()` flows). This will ensure dict-backed tests and `simulation` class flow work.
 2. Add a small `compat.py` to `src/emergent/salmon_abm/` that re-exports legacy API points (`simulation`, `PID_controller`, `summary`) pointing to the new modules when appropriate. Keep `sockeye.py` as a deprecated shim and prefer new imports.
 3. Defer `summary.kaplan_curve()` and related summary-heavy functions until after the sim runs; note in MIGRATION.md that Kaplan-Meier requires `sksurv` and will be re-enabled later.

- Tests & validation:
  - After the IO guard patches and `compat.py` shim, run `pytest -q src/emergent/salmon_abm/tests --basetemp=outputs/pytest_tmp` to validate that `simulation.run()` compiles and the test harness passes.

- Non-blocking enhancements (later):
  - Replace magic constants and hard-coded buffers with config-driven parameters.
  - Tune PID to be a function of length and water velocity (per TODO comments).
  - Performance: dataset compression, chunking, and memory layout optimization.

What I will do next (unless you direct otherwise):
- Apply IO guards for the most likely problematic direct `self.hdf5[...]` reads (notably `boundary_surface()` and any HDF reads used in `timestep()` paths). 
- Add `compat.py` with minimal re-exports.
- Run the `salmon_abm` tests and report results.

If that sounds good I will proceed now.
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

Remaining items from the attached plan (updated)
-----------------------------------------------
This project has moved past the initial scaffolding stage. The list below
focuses on active work (deprecated modules are intentionally ignored for now).

1) `utils.py` — pure helpers (geo_to_pixel, pixel_to_geo, shape/slice helpers,
  interpolation, front-mask helpers)
  - Status: IMPLEMENTED and unit-tested.

2) `io.py` — environment import, higher-level HDF5 helper wrappers, movie and
  Excel conveniences
  - Status: IMPLEMENTED. Added `safe_hdf5_open`, `write_sim_initial`,
    `enviro_load_many`, `longitudinal_chainage`, and `movie_frames_from_stack`.

3) `pid.py` — PID controller
  - Status: IMPLEMENTED and unit-tested (vectorized gains, robust `update`).

4) `simulation.py` — full simulation runner exposing `run()` and `timestep()`
  APIs and wiring behavior/fatigue/movement and HDF5 writes.
  - Status: PARTIAL. `simulation` has been expanded and now uses `io.write_sim_initial`
    for HDF5 setup; `timestep()` orchestration is present. The remaining work is
    to finalize the public `run()` API (PID plumbing, optional video hooks,
    clearer lifecycle management) and add focused tests.

5) `summary.py` — reporting & summary methods
  - Status: SKIPPED for now (deprioritized).

6) Compatibility shim (`compat.py`) to expose the old `sockeye` API
  - Status: NOT STARTED. `sockeye.py` currently acts as a deprecation shim.

7) Documentation updates and migration README entry
  - Status: PARTIAL. This `MIGRATION.md` is updated; further README/docs edits
    remain.

8) Test sweep: run full repository tests and address regressions
  - Status: NOT STARTED. Focused package tests for `salmon_abm` pass locally.

Notes
-----
- Deprecated modules (`src/emergent/salmon_abm/deprecated/*`) are intentionally
  excluded from the active migration work at this time.
- Compression and dataset optimization will be considered later during a
  dedicated performance pass; the current priority is API clarity and testable
  behavior.

How to run tests locally (Windows)
----------------------------------
Use a repository-local pytest basetemp to avoid Windows permission issues:

```powershell
pytest -q --basetemp=outputs/pytest_tmp
```

Next recommended step
---------------------
Finish expanding the `simulation` runner API (task #4). Specifically:

- Finalize `simulation.run()` so callers can:
  - configure PID tuning or pass a controller instance
  - select HDF5 write frequency and optional video hooks
  - receive a stable return value / status and error handling

- Add focused unit/integration tests that exercise `timestep()` and `run()`
  with small `num_agents` and `num_timesteps` using the dict-like HDF5 store.

Why this next
-------------
The core modules (`utils`, `io`, `pid`) are implemented and tested; making the
simulation runner complete will allow downstream tools and experiments to use
the new modular code with confidence and minimal duplication. Once the runner
API is stable we can add `compat.py` and then run a broader test sweep.

Suggested follow-up after `simulation` is complete:
- Add `compat.py` to centralize the deprecated API surface.
- Update docs and `MIGRATION.md` with usage examples for the new runner.
- Run full repository tests and prioritize any regressions exposed.

If you want I will proceed to finish the `simulation.run()` API and add
tests for it next.
