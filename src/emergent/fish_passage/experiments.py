"""
experiments.py

Preamble/Module plan for experiment orchestration and parameter sweeps (moved to fish_passage).

Responsibilities (planned):
- Define scenario templates, run parameter sweeps, and aggregate results into reproducible reports.
- Provide `Experiment` class to configure runs and `run_experiment` helper to execute batches.
- Export results to `outputs/` with standardized filenames and metadata.

Notes:
- Experiments should be deterministic given a seed; document required inputs in `scenarios/`.
"""
