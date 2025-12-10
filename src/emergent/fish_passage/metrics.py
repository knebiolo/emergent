"""
metrics.py

Preamble/Module plan for metrics computation and aggregation (moved to fish_passage).

Responsibilities (planned):
- Compute schooling metrics (cohesion, alignment, separation), energy statistics, passage success rates.
- Provide streaming aggregators for per-timestep and per-episode statistics.
- Export helpers to format metrics for viewer UI and for logging/CSV outputs.

Notes:
- Implement unit-tested pure functions for core calculations; avoid side-effects.
"""
