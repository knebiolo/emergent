"""
agents.py

Preamble/Module plan for agent definitions and lifecycle management.

Responsibilities (planned):
- Define `FishAgent` dataclass with physical properties (position, heading, body_length, energy, alive flag).
- Agent lifecycle: initialization, sensing API (query environment / neighbors), and action primitives (apply_thrust, set_heading).
- Keep agent methods small and side-effect minimal; simulation orchestration will manage stepping.

Notes:
- Avoid embedding environment access inside agent methods; use injected interfaces for environment queries to keep agents testable.
"""
