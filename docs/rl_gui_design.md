# RL GUI Design Document — Exact Reconstruction Plan

Purpose
- Provide a precise, actionable blueprint to reconstruct the original RL Training GUI (v1) into `salmon_viewer_v2.py` with pixel/behavioral parity where possible.

Source of truth
- `src/emergent/salmon_abm/salmon_viewer.py` is the canonical reference. The reconstruction must reproduce widget labels, layout ordering, control names, and behavioral side-effects documented below.

High-level layout
- Window title: `SalmonViewer` (exact text)
- Main container: horizontal layout with three columns:
  - Left column (tools and metrics, fixed narrow column)
  - Center (main display area: `PlotWidget` which may be replaced by `GLViewWidget` when GL is active)
  - Right column (playback, RL controls, toggles, status)

Left column (top-to-bottom)
- `PlotWidget` subsidiary controls: (in original, some metric plots and time-series were present; reproduce the compact metrics box)
- Metrics box (`QGroupBox` titled `Metrics`) with vertical layout containing labels in this exact order and text format:
  1. `Mean Speed: --`
  2. `Max Speed: --`
  3. `Mean Energy: --`
  4. `Min Energy: --`
  5. `Upstream Progress: --`
  6. `Mean Centerline: --`
  7. `Mean Passage Delay: --`
  8. `Passage Success: --`
  9. `Mean NN Dist: --`
 10. `Polarization: --`
- Per-episode metric checkboxes placed inline with labels. Label keys and their corresponding label widgets:
  - `collision_count` -> `Collision Count: --`
  - `mean_upstream_progress` -> uses `Upstream Progress` label
  - `mean_upstream_velocity` -> `Mean Upstream Velocity: --`
  - `energy_efficiency` -> uses `Mean Energy` label
  - `mean_passage_delay` -> uses `Mean Passage Delay` label
- Per-episode metrics plot: `Per-Episode Metrics` (pyqtgraph `PlotWidget`) with bottom axis labeled `Episode` and left axis `Metric Value`.

Center column
- Primary canvas: `pyqtgraph.PlotWidget` by default (600x600 min). When GL is available, replace with `pyqtgraph.opengl.GLViewWidget` in-place maintaining same visual area.
- If GL mesh is present, add `GLMeshItem` and `GLScatterPlotItem` for agents. Camera should be positioned to fit mesh extents.

Right column (top-to-bottom, exact ordering)
1. `Play` (QPushButton) — on click: set `paused=False` and update button states
2. `Pause` (QPushButton) — toggles pause state
3. `Reset` (QPushButton) — calls `reset_simulation()`; resets timestep, episode, reward counters
4. `Rebuild TIN` (QPushButton) — calls TIN builder and updates `last_mesh_payload`
5. `Toggle Perimeter` (QPushButton) — toggles visibility of perimeter overlay
6. `Vertical Exaggeration` group (`QGroupBox`) containing:
   - Label: `Z Exag: 1.00x` (updates live)
   - Slider: horizontal, range 1..500, default 100
7. Display toggles (checkboxes): `Show Dead`, `Show Direction`, `Show Tail` (in that order)
8. RL status labels (each a `QLabel`):
   - `Episode: 0 | Timestep: 0`
   - `Reward: 0.00`
   - `Best: 0.00`

Behavioral wiring / callbacks
- `update_simulation()` should use `sim.timestep(...)` and then call `update_displays()` and `update_rl_training()` (if `rl_trainer` present). The simulation owns PID controllers (no viewer fallback).
- `update_metrics_panel(metrics)` updates all metrics labels using numeric formatting `'{:.2f}'` where appropriate, and appends per-timestep values to episode accumulators when corresponding checkboxes are checked.
- `update_rl_training()` should compute reward using trainer methods, manage episode completion, mutate weights, save best weights to `outputs/rl_training/best_weights.json`, mutate trainer weights, and reset sim positions between episodes exactly as in the original.
- Per-episode plotting: when an episode completes, compute means from accumulators and update `per_episode_plot` series for each tracked metric.

Assets and visual parity
- Use Qt default styles but preserve ordering, spacing, and group titles. For closer pixel parity, capture screenshots from v1 and iterate with Qt stylesheet adjustments — request these screenshots if exact pixel matching is required.

Acceptance criteria
- Layout: left/center/right columns with widgets in exact ordering.
- Labels and default text must match exactly (see list above).
- Play/Pause/Reset/Rebuild/TIN/Toggle Perimeter must call the same functions and cause same side-effects (resetting sim state, saving weights, rebuilding TIN screenshot saved to outputs).
- `sim.perimeter_*` remains the single source of truth; viewer must not recompute perimeter for production.

Implementation plan
1. Create `docs/rl_gui_design.md` (this file) and get approval.
2. Implement strict widget tree in `salmon_viewer_v2.py` matching ordering and labelling.
3. Port `update_metrics_panel`, `update_rl_training`, and per-episode plotting from `salmon_viewer.py` line-by-line where possible, and refactor to keep logic testable.
4. Add headless checks and manual verification checklist.
5. Iterate visual fixes until acceptance criteria met.

Timeline & checkpoints
- Day 0: Design doc (this file) approved.
- Day 0-1: Implement widget tree and basic wiring (buttons + sliders + toggles).
- Day 1-2: Port metrics/training wiring and add headless checks.
- Day 2: Manual visual verification and final tweaks.

If you approve this design doc, I will proceed to port the remaining functions and ensure the UI matches exactly. If there are specific screenshots or behaviors to preserve beyond what's in the v1 code, attach them and I'll incorporate them into the acceptance criteria.
