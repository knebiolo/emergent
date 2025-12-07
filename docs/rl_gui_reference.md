# RL Training GUI Reference

This file documents the RL Training GUI widgets, labels, and layout as found in the original `salmon_viewer.py`. Use this as the authoritative reference to reproduce exact design, form, and function in `salmon_viewer_v2.py`.

Top-level window title:
- `SalmonViewer`

Main layout:
- Left: `PlotWidget` (pyqtgraph) used as primary display area; default size ~600x600.
- Right: vertical control panel with playback, toggles, metrics, and RL labels.

Playback controls (right panel, top):
- `Play` (QPushButton)
- `Pause` (QPushButton)
- `Reset` (QPushButton)

Visual controls / utilities:
- `Vertical Exaggeration` group:
  - Label: `Z Exag: 1.00x`
  - Slider: Horizontal slider (range 1..500, default 100)
- `Rebuild TIN` (button)
- `Toggle Perimeter` (button) [optional]

Display toggles (checkboxes):
- `Show Dead`
- `Show Direction`
- `Show Tail`

Metrics (compact group at bottom of right pane): labels include:
- `Mean Speed: --`
- `Max Speed: --`
- `Mean Energy: --`
- `Min Energy: --`
- `Upstream Progress: --`
- `Mean Centerline: --`
- `Mean Passage Delay: --`
- `Passage Success: --`
- `Mean NN Dist: --`
- `Polarization: --`

Per-episode metrics tracking (checkboxes inline with labels):
- `collision_count` (label: `Collision Count: --`)
- `mean_upstream_progress` (uses existing upstream label)
- `mean_upstream_velocity` (label: `Mean Upstream Velocity: --`)
- `energy_efficiency` (maps to `Mean Energy` label)
- `mean_passage_delay` (maps to `Mean Passage Delay` label)

Per-episode plot:
- `Per-Episode Metrics` (pyqtgraph PlotWidget), bottom of metrics group; x-axis `Episode`, y-axis `Metric Value`.

RL status labels (right panel):
- `Episode: 0 | Timestep: 0`
- `Reward: 0.00`
- `Best: 0.00`

Notes:
- Widget labels and ordering should match exactly; spacing should be similar (use QVBoxLayout and QHBoxLayout with small spacings).
- Default values (slider default, checkboxes unchecked) must match reference.
- Callback names and side-effects (e.g., `reset_simulation` should reset timestep, episode, and agents) should remain functionally identical.
