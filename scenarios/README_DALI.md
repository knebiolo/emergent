# MV Dali Bridge Collision Scenario

## Overview
Counterfactual analysis of the March 26, 2024 MV Dali collision with the Francis Scott Key Bridge in Baltimore.

**Research Question:** Could the 6 fatalities have been prevented?

## Historical Facts
- **Date/Time:** March 26, 2024, 01:07 EDT (Fort McHenry Channel entry)  01:28:44 EDT (collision)
- **Vessel:** MV Dali (IMO 9697428), 299.92m LOA, 48.2m beam, container ship
- **Power Loss:** 01:24:59 EDT (~17.5 minutes after channel entry, 1047 seconds in simulation)
- **Casualties:** 6 construction workers killed, 2 survivors
- **Cause:** Electrical failure (loose wire connection)

## Simulation Setup

### Configuration Files
- **scenarios/dali_bridge_collision_config.json** - Main scenario configuration
  - 30 minute duration (1800 seconds)
  - CBOFS environmental forcing (historical conditions)
  - Power loss at t=1047s
  - 5 parametric variations for counterfactual analysis

### Run Commands

**Test single baseline run:**
```powershell
python scripts/run_dali_scenario.py --config scenarios/dali_bridge_collision_config.json
```

**Batch mode (all 5 variations, headless):**
```powershell
python scripts/run_dali_scenario.py --config scenarios/dali_bridge_collision_config.json --batch --headless
```

**Single variation by ID:**
```powershell
python scripts/run_dali_scenario.py --config scenarios/dali_bridge_collision_config.json --run-id 2 --headless
```

## Parametric Variations

1. **Baseline** - Historical conditions (power loss at 1047s, t=0 departure)
2. **Early Failure** - Power loss at 900s (2.5 min earlier, further from bridge)
3. **Late Failure** - Power loss at 1200s (2.5 min later, closer to bridge)
4. **Early Tide** - Depart 60 min earlier (different tidal current state)
5. **Late Tide** - Depart 60 min later (different tidal current state)

## Outputs

Each run produces:
- **Trajectory CSV:** outputs/dali_scenario/run001_baseline_trajectory.csv
  - Columns: time_s, lon, lat, speed_knots, heading_deg
- **Collision metrics:** Automatically printed to console

Batch run produces:
- **Summary CSV:** outputs/dali_scenario/dali_batch_results.csv
  - All runs with collision yes/no, CPA, impact speed

## Next Steps
1. Test baseline run (verify collision reproduces)
2. Run batch overnight
3. Analyze sensitivity to power loss timing
4. Write paper section on counterfactual scenarios
