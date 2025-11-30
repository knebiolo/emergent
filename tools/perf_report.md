# Performance report

Summary of profiler runs (2000 agents):

- Pre-merge (before merging swim/drag/fatigue kernels):
  - `tools/profile_2000_premerge.pstats` — `timestep` cumulative ≈ 0.545 s (50 timesteps run)

- Post-merge (after merging into `_merged_swim_drag_fatigue_numba`):
  - `tools/profile_2000_postmerge.pstats` — `timestep` cumulative ≈ 0.517 s (50 timesteps run)

- Net change: post-merge is ~5% faster on `timestep` cumulative time vs pre-merge.

Thread sweep (short timed runs, 2000 agents × 20 timesteps):

- threads=1: `tools/profile_2000_t1.pstats` — timestep cumtime = 0.240 s
- threads=2: `tools/profile_2000_t2.pstats` — timestep cumtime = 0.193 s
- threads=4: `tools/profile_2000_t4.pstats` — timestep cumtime = 0.185 s  (best)
- threads=8: `tools/profile_2000_t8.pstats` — timestep cumtime = 0.199 s

Recommendation

- Use `NUMBA_NUM_THREADS=4` and `NUMBA_DEFAULT_NUM_THREADS=4` for best throughput on this machine.
- Always run a small warmup before timed production runs to ensure numba caches are populated:

  ```powershell
  python tools\numba_warmup.py
  $env:NUMBA_NUM_THREADS=4
  $env:NUMBA_DEFAULT_NUM_THREADS=4
  ```

Files created/modified by this optimization effort

- Modified: `src/emergent/salmon_abm/sockeye_SoA.py` — added `_merged_swim_drag_fatigue_numba`, updated call sites.
- Added: `tools/numba_warmup.py`
- Added: `tests/test_perf_regression.py` (lightweight check that the profiler completes)
- Added: `.github/workflows/perf-benchmark.yml` (CI warmup + short benchmark)
- Updated: `RUNNING.md` with Numba warmup and recommended env settings.
- Generated: `tools/perf_report.md` (this file) and multiple pstats under `tools/`.

Notes about running tests locally

- I attempted to run `pytest` but the local environment raised a Windows fatal exception originating in `h5py`/`h5netcdf`/`pyqtgraph` imports. This looks like an environment/extension issue on this machine (C-extension load failure). Recommended mitigations:
  - Run tests in a clean headless environment (minimal deps) or use a CI runner (Ubuntu) where the GitHub Actions job will run the perf benchmark without GUI deps.
  - To avoid GUI-heavy tests locally, run a targeted subset, or run `pytest -k "not gui and not heavy"` (you may need to tag tests appropriately).

Next steps I can take (pick one):
- Merge `_calc_battery_numba` into the merged kernel and re-profile (may yield small additional gains).
- Prepare a PR description and CI checks to push these changes upstream.
- Draft a short CHANGELOG entry and finalize the performance report with exact numbers extracted from pstats (if you want more formal benchmarking output).

If you want, I can prepare the PR and include the warmup script and CI workflow in it. Also tell me if you want me to proceed with the optional battery-kernel merge now.
