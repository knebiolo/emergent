# RL Port Plan: `origin/opengl_pyqtgraph` -> `fish_passage_module`

Date: 2026-03-06

## Objective

Port the useful RL capabilities from `origin/opengl_pyqtgraph` into the current split-module architecture without reintroducing the legacy monolith.

## Deep-Dive Findings

### 1) Current branch RL stack is modular but has schema drift

Primary files on `fish_passage_module`:

- `src/emergent/salmon_abm/rl_training.py`
- `src/emergent/salmon_abm/rl_training_viewer.py`
- `tools/train_behavioral_weights.py`
- `src/emergent/salmon_abm/simulation.py`

Observed issues:

- `BehavioralWeights.to_test_weights_dict()` references `separation_weight`, but the dataclass does not define it.
- Confirmed runtime failure:
  - `AttributeError: 'BehavioralWeights' object has no attribute 'separation_weight'`
- Test/document expectations are out of sync with the current dataclass (for example, references to `sensory_range`).

### 2) Current reward function is flow-integration-first and omits some legacy signals

`compute_episode_reward(...)` currently emphasizes:

- schooling (cohesion/alignment/separation),
- upstream progress via flow-vector integration,
- energy efficiency,
- fatigue/stagnation/rheotaxis penalties.

Not currently implemented in the same explicit way as legacy:

- explicit collision event penalty (count-based),
- explicit dry/shallow penalty from depth states,
- drafting benefit is still a placeholder (`0.0`).

### 3) Legacy branch RL logic exists, but in monolithic `sockeye.py`

Key legacy sources:

- `origin/opengl_pyqtgraph:src/emergent/salmon_abm/sockeye.py`
- `origin/opengl_pyqtgraph:src/emergent/salmon_abm/hecras_helpers.py`

Useful legacy capabilities to salvage:

- HECRAS centerline derivation helpers:
  - `derive_centerline_from_hecras_distance(...)`
  - `extract_centerline_fast_hecras(...)`
- reward components for collision and dry/shallow penalties,
- drafting computation path (heavier implementation).

### 4) Centerline helper module was deleted in current branch

Cross-branch diff shows:

- `src/emergent/salmon_abm/hecras_helpers.py` was removed.
- No equivalent centerline extraction helper currently exists in active modules.

### 5) Training entrypoints are still raster-oriented

- `tools/train_behavioral_weights.py` discovers `depth.tif`, `vel_x.tif`, etc.
- `rl_training_viewer.py` hard-fails without `longitudinal.shp`.
- This conflicts with the desired HECRAS-direct workflow.

## Port Strategy

Do not cherry-pick legacy `sockeye.py` commits directly. Manually port targeted logic into current modules:

- keep `simulation.py` + `behavior.py` + `rl_training.py` architecture,
- add only reusable helper logic,
- keep legacy viewer/runtime code out.

## Phased Plan

### Phase 0: Baseline RL integrity (required first)

Scope:

- fix `BehavioralWeights` schema mismatch (`to_test_weights_dict` compatibility),
- align RL docs/tests with actual dataclass fields,
- add/refresh a small RL smoke command.

Acceptance criteria:

- `BehavioralWeights().to_test_weights_dict()` does not raise.
- RL smoke run can initialize trainer and execute at least one episode.

### Phase 1: Centerline and longitudinal fallback for HECRAS-direct

Scope:

- create `src/emergent/salmon_abm/hecras_centerline.py` (port subset from legacy `hecras_helpers.py`),
- implement fallback behavior:
  - if `longitudinal_profile` is missing and `hecras_plan_path` exists, derive centerline/longitudinal geometry automatically,
- persist derived geometry (for reproducibility/debugging).

Acceptance criteria:

- RL training can start with HECRAS plan + start polygon, without requiring `longitudinal.shp`.
- Derived centerline exists and is non-empty for the canonical Nuyakuk plan.

### Phase 2: Reward parity upgrades (targeted, not monolithic)

Scope:

- extend `compute_episode_reward(...)` to support optional:
  - collision penalties,
  - shallow/dry penalties,
  - drafting proxy (lightweight first pass),
- add the required episode history channels (for example depth history) in `RLTrainer.run_episode`.

Acceptance criteria:

- reward component dict includes new terms when data is provided,
- synthetic tests confirm sign and magnitude behavior:
  - more collisions -> lower reward,
  - more shallow/dry exposure -> lower reward.

### Phase 3: Training entrypoints and viewer alignment

Scope:

- update `tools/train_behavioral_weights.py` to support HECRAS-direct inputs (mirror production script style),
- update `rl_training_viewer.py` to allow auto-derived longitudinal instead of hard-failing on missing shapefile,
- preserve raster mode as fallback.

Acceptance criteria:

- CLI training works in both modes:
  - HECRAS direct,
  - raster inputs.
- viewer can launch RL training in HECRAS-direct mode without manual longitudinal shapefile.

### Phase 4: Validation and hardening

Scope:

- add/refresh tests for:
  - centerline fallback,
  - reward components,
  - minimal end-to-end RL episode in HECRAS-direct mode,
- update docs to reflect canonical RL workflow.

Acceptance criteria:

- test subset for RL/centerline passes in the project test env,
- docs include exact command lines for both modes.

## Recommended Port Inputs (Legacy References)

Use these as implementation references only (manual extraction):

- `origin/opengl_pyqtgraph:src/emergent/salmon_abm/hecras_helpers.py`
- `origin/opengl_pyqtgraph:src/emergent/salmon_abm/sockeye.py` (RLTrainer metrics/reward sections)

Notable historical commits for helper logic context:

- `1b15ef4` (wetted perimeter + helper additions)
- `62982a9` (centralized HECRAS perimeter/mapping logic)
- `c52bb0e` (KDTree defensive behavior)

## Risks and Mitigations

Risk:

- centerline quality in braided reaches can be noisy.

Mitigation:

- keep flow-integration metric as default baseline and add centerline metric as optional/fallback or blended signal.

Risk:

- collision/drafting metrics can become expensive at high N.

Mitigation:

- use KDTree and configurable sampling cadence for expensive components.

Risk:

- interface churn from adding more episode history channels.

Mitigation:

- move episode output from raw tuple to a named structure (dataclass/dict) with backward-compatible adapters.

## Proposed Delivery Order (PR-sized)

1. RL schema cleanup + tests/doc sync.
2. HECRAS centerline helper + longitudinal fallback.
3. Reward parity extensions (collision/shallow + optional drafting proxy).
4. Training CLI/viewer HECRAS-direct support + final docs.

