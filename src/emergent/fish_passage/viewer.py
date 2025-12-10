"""
viewer.py

Preamble/Module plan for the Salmon ABM viewer components (moved to fish_passage).

This file will contain the main `SalmonViewer` QWidget and helper UI wiring.
Responsibilities (planned):
- High-level `SalmonViewer` QWidget class that composes rendering, controls, and metrics panes.
- `launch_viewer(simulation, ...)` convenience function to initialize QApplication and run the viewer.
- Lightweight UI event loop handling, timers for simulation stepping, and integration points for RL training visualization.
- Minimal defensive handling at public boundaries; internal functions use assertions and explicit validation.

Notes:
- Rendering specifics will be delegated to `rendering.py`.
- Heavy GL mesh building will be performed in `rendering.py` worker threads to keep viewer responsive.
- All user-visible UI changes require explicit approval per `long_term_ui_policy`.
"""
