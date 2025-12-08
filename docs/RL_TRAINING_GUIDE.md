# Reinforcement Learning for Behavioral Weight Optimization

## Overview

This RL system trains **instinctual behavioral parameters** that produce realistic schooling and upstream migration in the salmon ABM. The key insight: separate **persistent instincts** (learned once, reused forever) from **ephemeral spatial knowledge** (reset each simulation).

## Architecture

### Persistent Instincts (Learned & Frozen)
These behavioral weights control emergent behavior and are learned once through RL training:

- **Schooling dynamics**
  - `cohesion_weight`: Attraction to group center
  - `alignment_weight`: Tendency to match neighbors' heading
  - `separation_weight`: Short-range repulsion from neighbors
  - `separation_radius`: Distance for separation activation

- **Environmental responses**
  - `rheotaxis_weight`: Swim against flow (upstream migration)
  - `border_cue_weight`: Avoid channel edges/banks
  - `border_threshold_multiplier`: When to activate border avoidance
  - `border_max_force`: Cap on border repulsion

- **Collision avoidance**
  - `collision_weight`: Repulsion from other agents
  - `collision_radius`: Distance for collision detection

### Ephemeral Spatial State (Reset Each Simulation)
These are reset at the start of each new simulation:

- Agent positions (X, Y, Z)
- Velocities and headings
- Energy/battery levels
- Memory buffers (eddy escape, past positions)
- Dead/alive status
- Route memory through specific geometry

## Usage

### 1. Train Behavioral Weights (One-Time)

```powershell
# Train for 50 episodes with 100 timesteps each, 200 agents
python tools/train_behavioral_weights.py --episodes 50 --timesteps 100 --agents 200

# Trained weights saved to: outputs/rl_training/behavioral_weights.json
```

**Training Process:**
1. Initialize simulation with default behavioral weights
2. Run episode (100 timesteps)
3. Compute reward based on:
   - Cohesive schooling (agents stay together at ideal distance)
   - Upstream progress (net forward movement)
   - Velocity alignment (coordinated swimming)
   - Boundary avoidance (stay in channel)
   - Low collision rate
   - Energy efficiency (smooth movement)
   - Survival (minimize mortality)
4. Mutate weights slightly for exploration
5. Reset spatial state (positions, velocities) but keep behavioral weights
6. Repeat, saving best weights

### 2. Use Trained Weights in Simulations

```python
from emergent.salmon_abm.sockeye import simulation

# Initialize simulation
sim = simulation(**config)

# Load pre-trained behavioral weights
sim.load_behavioral_weights('outputs/rl_training/behavioral_weights.json')

# Run simulation - agents exhibit learned schooling/migration behavior
for t in range(num_timesteps):
    sim.timestep(t, dt=1.0, gravity=9.81, pid_controller=pid)
```

### 3. Reset Spatial State Between Runs

```python
# First simulation run
for t in range(100):
    sim.timestep(t, 1.0, 9.81, pid)

# Reset for new run (preserves behavioral weights, resets positions/memory)
sim.reset_spatial_state()

# Second simulation run - same behavior, fresh spatial state
for t in range(100):
    sim.timestep(t, 1.0, 9.81, pid)
```

## Reward Function

The reward function shapes emergent behavior:

```python
reward = (
    cohesion_quality * 1.0         # Tight groups at ideal spacing
    + upstream_progress * 2.0      # Net forward movement
    + alignment_quality * 0.5      # Coordinated swimming
    - boundary_violations * 10.0   # Penalty for edge proximity
    - collisions * 5.0             # Penalty for agent overlap
    + movement_smoothness * 0.5    # Energy-efficient trajectories
    - mortality * 100.0            # Strong penalty for deaths
)
```

## Key Files

- **`src/emergent/salmon_abm/sockeye.py`** (formerly `sockeye_SoA_OpenGL_RL.py`): Main simulation with RL infrastructure
  - `BehavioralWeights`: Container for instinctual parameters
  - `RLTrainer`: Training loop with reward computation
  - `simulation.apply_behavioral_weights()`: Apply learned weights
  - `simulation.reset_spatial_state()`: Reset ephemeral state
  
- **`tools/train_behavioral_weights.py`**: Training script
  - Sets up HECRAS-based simulation
  - Runs RL training loop
  - Saves best weights to JSON

- **`outputs/rl_training/behavioral_weights.json`**: Trained weights (generated)

## Benefits

1. **Realistic emergent behavior**: Schooling and navigation arise from learned instincts, not hardcoded rules
2. **Transferable across geometries**: Train once, use on any river layout
3. **No spatial "cheating"**: Agents don't memorize optimal routes between runs
4. **Tunable**: Easy to adjust reward function to emphasize different behaviors
5. **Data-driven**: Weights optimized for actual performance metrics

## Next Steps

1. **Tune reward function**: Adjust weights to match observed salmon behavior
2. **Multi-objective optimization**: Balance competing goals (speed vs. energy)
3. **Curriculum learning**: Train on simple geometries first, then complex ones
4. **Population heterogeneity**: Learn distributions of weights for individual variation
5. **Online adaptation**: Allow limited learning during simulation (tactical only, not spatial)

## Example Workflow

```powershell
# 1. Train behavioral weights (one-time, ~30 minutes)
python tools/train_behavioral_weights.py --episodes 100 --timesteps 200 --agents 500

# 2. Run production simulations with learned weights
python tools/run_hecras_opengl.py --agents 1000 --timesteps 500 --weights outputs/rl_training/behavioral_weights.json

# 3. Test on new geometry (weights transfer automatically)
python tools/run_hecras_opengl.py --hecras-plan data/new_river.p05.hdf --agents 1000 --weights outputs/rl_training/behavioral_weights.json
```

## Technical Details

### Training Algorithm
- **Strategy**: Evolutionary / policy gradient hybrid
- **Exploration**: Small Gaussian perturbations to weights each episode
- **Selection**: Keep best-performing weights across all episodes
- **Episodes**: 50-100 recommended for convergence
- **Timesteps per episode**: 100-200 (balance training time vs. behavior quality)

### Computational Cost
- **Training**: ~1-5 minutes per episode (200 agents, 100 timesteps)
- **Total training time**: 1-8 hours for 100 episodes
- **Production inference**: No overhead (weights are just static multipliers)

### Memory Requirements
- Training simulation: ~2-4 GB RAM
- Saved weights: <1 KB (JSON file)

---

**Questions?** See the code comments in `src/emergent/salmon_abm/sockeye.py` (legacy name: `sockeye_SoA_OpenGL_RL.py`) for implementation details.
