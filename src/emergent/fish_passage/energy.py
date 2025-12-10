"""
energy.py

Preamble/Module plan for energy budget and metabolism models (moved to fish_passage).

Responsibilities (planned):
- Model per-agent energy budget, metabolic consumption rates, and recovery mechanisms.
- Implement functions:
  - `compute_metabolic_costs(agent, speed, mode)`
  - `apply_energy_update(agent, cost, dt)`
  - `estimate_time_to_fatigue(agent)`
- Provide unit-testable, pure functions for core calculations; side-effects minimal and explicit.

Notes:
- Energy constants and tuning parameters live in `config.py`.
"""
