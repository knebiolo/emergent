# Salmon ABM Paper Experiment Suite (Canonical)

Date: 2026-03-07  
Branch context: `fish_passage_module`

## Research Questions

1. How does enabling schooling cues change movement outcomes versus disabling schooling cues?
2. How does population size affect passage delay?

## Core Hypotheses

- H1: Schooling-enabled runs produce lower delay (faster upstream progress) than no-schooling runs.
- H2: Delay increases with population size (congestion effect).
- H3: Schooling benefit is population-dependent (interaction between schooling condition and `N`).

## Canonical Runtime Setup (Spark)

Hold these constant across all experiments:

- Mode: HECRAS direct (`--hecras-plan`)
- `dt`: `1.0`
- `nsteps`: `3600` (1 hour simulated time)
- HECRAS mapping: `--hecras-time-mode loop --hecras-k 8`
- Start polygon: `data/salmon_abm/shapes/start_loc_river_right.shp`
- Longitudinal profile: `data/salmon_abm/shapes/longitudinal.shp`
- Runner: `tools/test_salmon_abm.py` (chosen for explicit `--seed`)
- Output backend: async process + video-lite keys (`X`, `Y`, `battery`, `heading`)

## Experimental Factors

- Factor A (Schooling):
  - `schooling_on`: default behavioral weights
  - `schooling_off`: `cohesion_weight=0`, `alignment_weight=0`, all other defaults unchanged
- Factor B (Population size):
  - `N ∈ {250, 500, 1000, 2000, 5000, 10000}`
- Replicates:
  - Seeds `1..10` per condition

Total runs: `2 * 6 * 10 = 120`.

## Schooling Condition Files

Create explicit weight files so both conditions are reproducible:

```bash
cd /home/kevinnebiolo/emergent
conda run -n emergent python -c "import json; from pathlib import Path; from emergent.salmon_abm.rl_training import BehavioralWeights; out=Path('outputs/experiment_weights'); out.mkdir(parents=True, exist_ok=True); on=BehavioralWeights().to_dict(); off=BehavioralWeights().to_dict(); off['cohesion_weight']=0.0; off['alignment_weight']=0.0; json.dump(on, open(out/'schooling_on_default.json','w'), indent=2); json.dump(off, open(out/'schooling_off_no_cohesion_alignment.json','w'), indent=2)"
```

## Single-Run Template

```bash
cd /home/kevinnebiolo/emergent
conda run -n emergent python tools/test_salmon_abm.py \
  --nagents 2000 \
  --nsteps 3600 \
  --dt 1.0 \
  --seed 1 \
  --hecras-plan /home/kevinnebiolo/emergent/data/salmon_abm/Nuyakuk_Production_.p08.hdf \
  --start-polygon /home/kevinnebiolo/emergent/data/salmon_abm/shapes/start_loc_river_right.shp \
  --longitudinal-profile /home/kevinnebiolo/emergent/data/salmon_abm/shapes/longitudinal.shp \
  --test-weights-file /home/kevinnebiolo/emergent/outputs/experiment_weights/schooling_on_default.json \
  --output-backend process \
  --output-keys video \
  --skip-trace \
  --skip-analysis \
  --outdir /home/kevinnebiolo/emergent/outputs/paper_experiments \
  --model-name paper_schooling_on_n2000_seed1
```

## Batch Execution Loop (Bash)

```bash
cd /home/kevinnebiolo/emergent
for schooling in on off; do
  if [ "$schooling" = "on" ]; then
    weights="/home/kevinnebiolo/emergent/outputs/experiment_weights/schooling_on_default.json"
  else
    weights="/home/kevinnebiolo/emergent/outputs/experiment_weights/schooling_off_no_cohesion_alignment.json"
  fi
  for n in 250 500 1000 2000 5000 10000; do
    for seed in 1 2 3 4 5 6 7 8 9 10; do
      conda run -n emergent python tools/test_salmon_abm.py \
        --nagents "$n" \
        --nsteps 3600 \
        --dt 1.0 \
        --seed "$seed" \
        --hecras-plan /home/kevinnebiolo/emergent/data/salmon_abm/Nuyakuk_Production_.p08.hdf \
        --start-polygon /home/kevinnebiolo/emergent/data/salmon_abm/shapes/start_loc_river_right.shp \
        --longitudinal-profile /home/kevinnebiolo/emergent/data/salmon_abm/shapes/longitudinal.shp \
        --test-weights-file "$weights" \
        --output-backend process \
        --output-keys video \
        --skip-trace \
        --skip-analysis \
        --outdir /home/kevinnebiolo/emergent/outputs/paper_experiments \
        --model-name "paper_schooling_${schooling}_n${n}_seed${seed}"
    done
  done
done
```

## Delay Definition (Primary Response Variable)

Define upstream progress from the longitudinal polyline:

- Let `L` be the longitudinal `LineString`.
- Let `s_i(t) = L.project(Point(x_i(t), y_i(t)))` in meters.
- Let `Δs_i(t) = s_i(t) - s_i(0)`.

Define delay-to-threshold for each agent:

- Choose threshold distance `D* = 250 m` (primary).
- `τ_i = min{ t : Δs_i(t) >= D* } * dt`.
- If never reached by end of run, mark as right-censored at `T = nsteps * dt`.

Run-level metrics:

- `median_delay_250m` (seconds)
- `reach_rate_250m` = fraction of agents reaching `D*` by `T`
- `median_progress_rate` = median `Δs_i(T) / T`

Secondary thresholds for sensitivity: `D* = 100 m`, `500 m`.

## Statistical Analysis Plan

- Primary model (run-level):
  - `delay ~ schooling + log(N) + schooling:log(N)`
- Also report nonparametric summaries:
  - Median and bootstrap 95% CI by condition (`schooling`, `N`)
  - Pairwise schooling-on vs schooling-off at each `N`
- For censored delay robustness:
  - Kaplan-Meier curves by condition
  - Log-rank tests (schooling effect within each `N`)

## Quality Controls

- Verify each run writes expected shape in `agent_data/X` and `agent_data/Y`.
- Confirm run metadata from model names (`schooling`, `N`, `seed`) matches file contents.
- Keep all forcing/config fixed except experimental factors.
- Do not mix raster and HECRAS direct runs in the same statistical model.

## Expected Compute Budget (Spark, rough)

Observed reference: `N=10000`, `nsteps=3600` completed in ~41 minutes.  
A full 120-run matrix is expected to take on the order of 1-2 node-days unless parallelized.

## Deliverables for Paper

- `outputs/paper_experiments/*.h5` (raw trajectories)
- `outputs/paper_experiments_metrics.csv` (run-level metrics)
- Figure set:
  - Delay vs population size, split by schooling condition
  - Reach-rate vs population size
  - Survival/delay curves (if censored analysis included)
- Methods text should cite:
  - fixed HECRAS direct forcing
  - seeded replicate design
  - delay metric definition from longitudinal projection
