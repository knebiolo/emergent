"""
movement.py

Preamble/Module plan for movement dynamics and kinematics (moved to fish_passage).

Responsibilities (planned):
- Low-level kinematics mapping thrust/heading to velocity updates.
- Turning dynamics, collision avoidance primitives, and position integrators.
- Functions:
  - `compute_velocity(agent, thrust, dt)`
  - `integrate_position(agent, velocity, dt)`
  - `resolve_collisions(agents, min_sep)`

Notes:
- Keep loops bounded and use vectorized ops where possible for performance.
"""
