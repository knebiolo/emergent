Tools README

- `profile_timestep_cprofile.py`: simple CI-friendly profiling script used by `.github/workflows/perf-benchmark.yml` to generate `tools/ci_profile.pstats` for basic performance benchmarking. It runs a few lightweight triangulations to simulate workload without requiring heavy datasets.
- `tin_smoke_test.py`: standalone TIN smoke test (developer use).

If CI still fails with missing artifacts, ensure that the `tools/` directory is present in the repository root and that Actions checks out the correct branch.