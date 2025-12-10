"""
behaviors.py

Preamble/Module plan for fish behavioral primitives and composites.

Responsibilities (planned):
- Implement behavioral building blocks: `swim_upstream`, `avoid_obstacle`, `school_with_neighbors`, `explore`.
- Behavioral controller that composes primitives into higher-level decisions based on sensor inputs.
- Keep functions pure where possible; side-effects limited to returning desired velocity/heading changes.

Notes:
- RL trainer will operate at a higher level, providing reward signals and optionally overriding behaviors for training episodes.
"""
