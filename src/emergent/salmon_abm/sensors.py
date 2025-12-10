"""
sensors.py

Preamble/Module plan for perception and sensing primitives.

Responsibilities (planned):
- Sampling environmental fields (depth, flow) at agent positions with optional noise.
- Neighbor sensing utilities (KDTree wrappers) and field-of-view selection.
- Functions:
  - `sample_environment(simulation, agent_pos)`
  - `query_neighbors(positions, radius)`
  - `simulate_sensor_noise(value, noise_model)`

Notes:
- Expose efficient batch operations to avoid per-agent Python loops where possible.
"""
