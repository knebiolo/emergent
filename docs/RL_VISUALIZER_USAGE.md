# RL Training Visualizer Usage Guide

## Overview

The RL Training Visualizer is a real-time PyQt5 application for watching behavioral weight optimization in progress. It provides a three-panel interface to monitor training, visualize agent movements, and control training parameters.

Current input constraints:
- Raster-first workflow (`depth.tif`, `vel_*.tif`) is required.
- `longitudinal.shp` is required in `--model-dir`.
- HECRAS-direct training/viewer input is not wired into this tool yet.

## Installation

The visualizer requires PyQt5:

```bash
pip install PyQt5
```

## Quick Start

Launch the visualizer with your model configuration:

```bash
python -m emergent.salmon_abm.rl_training_viewer \
  --model-dir data/salmon_abm \
  --start-polygon data/salmon_abm/start_loc_river_right.shp
```

Or use it programmatically:

```python
from emergent.salmon_abm.rl_training_viewer import RLTrainingViewer
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
viewer = RLTrainingViewer(
    model_dir="data/salmon_abm",
    start_polygon="data/salmon_abm/start_loc_river_right.shp"
)
viewer.show()
sys.exit(app.exec_())
```

## Interface Layout

### Left Panel: Behavioral Weights & Diagnostics

Displays:
- **Current Weights**: All 20 BehavioralWeights parameters with live updates
- **Training Diagnostics**: 
  - Current episode / Total episodes
  - Current reward
  - Best reward achieved
  - Improvement from initial weights
- **Reward Components**: Breakdown of reward calculation
  - Upstream progress
  - Schooling cohesion
  - Energy efficiency
  - Survival rate
  - Drafting benefits

### Center Panel: Simulation Visualization

Real-time OpenGL rendering of agent positions:
- Blue dots represent individual fish
- Updates after each episode completes
- Shows final positions from most recent episode
- Auto-scales to fit all agents
- Gray background for water areas

### Right Panel: Training Controls

**Playback Controls:**
- ▶ **Start Training**: Begin RL training with current parameters
- ⏸ **Pause**: Pause training (resume button appears)
- ⏹ **Stop**: Terminate training early

**Training Parameters:**
- **Episodes**: Number of training episodes (1-1000, default 50)
- **Timesteps**: Simulation steps per episode (10-1000, default 100)
- **Agents**: Fish per episode (10-1000, default 200)
- **Exploration**: Mutation noise level (0.01-1.0, default 0.1)

**Progress Monitoring:**
- Progress bar shows episode completion
- Status label shows current activity
- Training log shows detailed output

## Training Workflow

1. **Configure Parameters**: Set episodes, timesteps, agents, exploration noise
2. **Start Training**: Click "▶ Start Training" button
3. **Monitor Progress**:
   - Watch left panel for weight evolution
   - Watch center panel for agent behavior
   - Watch right panel for episode progress
4. **Control Execution**: Use pause/resume/stop as needed
5. **Review Results**: Check final weights and reward improvement

## Training Output

The visualizer runs training in a background thread and updates all panels in real-time:

**Episode Start:**
- Status updates to "Running episode X/Y"
- Progress bar advances

**Episode Complete:**
- Weights panel updates with best weights found so far
- Center canvas shows final agent positions
- Log shows reward and "✓ BEST" marker if new best found
- Reward components breakdown updates

**Training Complete:**
- Status shows "Training complete"
- Log shows summary:
  - Best reward achieved
  - Initial vs final reward
  - Total improvement
- Best weights remain displayed for export

## Configuration

### Command Line Arguments

```bash
--model-dir PATH          # Required: Path to environment files directory
--start-polygon PATH      # Required: Path to starting polygon shapefile
--model-name NAME         # Optional: Model name (default: salmon_abm)
--basin NAME              # Optional: Basin name (default: nuyakuk)
```

### Environment Files Required

The `--model-dir` must contain:
- `depth.tif`: Water depth raster (meters)
- `vel_x.tif`: X-velocity component (m/s)
- `vel_y.tif`: Y-velocity component (m/s)
- `vel_mag.tif`: Velocity magnitude (m/s)
- `vel_dir.tif`: Velocity direction (radians)
- `longitudinal.shp`: Longitudinal profile shapefile (required by current reward/path-progress logic)

### Start Polygon

Shapefile defining fish starting locations. Must be in same CRS as environment files (EPSG:26905 for Alaska).

## Advanced Usage

### Programmatic Control

```python
from emergent.salmon_abm.rl_training_viewer import RLTrainingViewer
from emergent.salmon_abm.rl_training import BehavioralWeights

# Create viewer
viewer = RLTrainingViewer(
    model_dir="data/salmon_abm",
    start_polygon="data/salmon_abm/start_loc_river_right.shp"
)

# Show window
viewer.show()

# Access training state after completion
if viewer.trainer:
    best_weights = viewer.trainer.best_weights
    best_reward = viewer.trainer.best_reward
    history = viewer.trainer.episode_history
    
    # Save best weights
    best_weights.to_json("outputs/best_weights.json")
```

### Custom Initial Weights

Modify `BehavioralWeights()` in the visualizer before training:

```python
from emergent.salmon_abm.rl_training import BehavioralWeights

# Load from previous training
initial_weights = BehavioralWeights.from_json("outputs/previous_best.json")

# Or customize manually
initial_weights = BehavioralWeights(
    cohesion_weight=1500.0,
    alignment_weight=30000.0,
    rheotaxis_weight=20000.0,
    # ... other parameters
)
```

### Tuning Exploration Noise

- **High noise (0.3-1.0)**: Wide exploration, may find better solutions but slower convergence
- **Medium noise (0.1-0.3)**: Balanced exploration/exploitation (recommended)
- **Low noise (0.01-0.1)**: Fine-tuning near local optimum

Start with default 0.1, increase if stuck in poor local optimum.

## Troubleshooting

### "ERROR: Missing configuration!"

Provide both `--model-dir` and `--start-polygon` arguments when launching.

### "ERROR: Model directory not found"

Check that path to model directory is correct and contains environment TIF files.

### "Longitudinal profile shapefile required but not found"

Add `longitudinal.shp` (and sidecar files `.shx`, `.dbf`, `.prj`) to `--model-dir`.

### "ERROR: Start polygon not found"

Verify shapefile path exists and includes .shp, .shx, .dbf, .prj files.

### Training hangs or freezes

- Check log panel for error messages
- Reduce number of agents or timesteps for faster episodes
- Ensure environment files are valid GeoTIFF rasters

### No agent visualization

- Training may still be running (check status label)
- Agents may be outside visible bounds (auto-scaling on next update)
- Check log for simulation errors

### Poor reward scores

- Try adjusting exploration noise
- Increase number of episodes for more thorough search
- Check initial weights are reasonable (see `BehavioralWeights` defaults)

## Performance Tips

1. **Start small**: Test with 3 episodes, 20 timesteps, 10 agents
2. **Scale up**: Once working, increase to 50 episodes, 100 timesteps, 200 agents
3. **Production runs**: 100-500 episodes with full timesteps (100-200)
4. **Use compute-only mode**: Visualizer already disables HDF5 writes for speed

## Integration with Training Scripts

The visualizer uses the same RL infrastructure as `tools/train_behavioral_weights.py`:

- Same `BehavioralWeights` dataclass
- Same `RLTrainer` class
- Same reward calculation
- Same simulation factory pattern

Results from visualizer can be used with command-line training and vice versa.

## Future Enhancements

Planned features:
- Real-time reward plot (reward vs episode)
- Weight evolution plot (show parameter changes over time)
- Episode replay controls (scrub through timesteps)
- Multi-threaded episode execution (parallel evaluations)
- Weight sensitivity analysis visualization
- Export training history to CSV/JSON

## Related Documentation

- [RL Training Guide](RL_TRAINING_GUIDE.md): Complete RL system overview
- [Biological Schooling Metrics](BIOLOGICAL_SCHOOLING_METRICS.md): Reward function details
- [User Config Options](USER_CONFIG_OPTIONS.md): Simulation parameters

---

**Created:** 2026-01-03  
**Last Updated:** 2026-03-06  
**Version:** 1.1
