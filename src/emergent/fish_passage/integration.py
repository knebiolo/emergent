"""
integration.py

Preamble/Module plan for glue code to run simulation + RL + viewer (moved to fish_passage).

Responsibilities (planned):
- Provide `run_headless(sim_config)` and `run_with_viewer(sim_config)` helpers that wire simulation, RL trainer, and viewer.
- Keep orchestration logic minimal and explicit; delegate heavy lifting to modules defined above.

Notes:
- Orchestration should maintain clear boundaries and provide hooks for testing and checkpointing.
"""
