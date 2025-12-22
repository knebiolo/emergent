Configuration options worth exposing in the Salmon ABM viewer

- death_on_dry_immediate (bool): If True (default), agents on dry ground are marked dead immediately. Otherwise they die after `time_out_of_water_threshold` timesteps.
- time_out_of_water_threshold (int): Number of timesteps allowed out of water before death when immediate-death is False. Default: 5.
- initial_velocity_jitter (float): Standard deviation (m/s) for per-agent initial water-relative velocity jitter. Default: 0.05.
- initial_sog_jitter_fraction (float): Fractional jitter applied to `ideal_sog` on reset (e.g. 0.1 => ±10%). Default: 0.1.
- strong_initial_sog_jitter (bool): When True, use `strong_initial_sog_jitter_fraction` instead of `initial_sog_jitter_fraction`.
- strong_initial_sog_jitter_fraction (float): Fraction used when strong jitter is enabled (e.g. 0.3).
- collision_penalty_per_event (float): Reward penalty per collision event applied during RL training. Default: 100.0.
- dry_penalty_per_agent (float): Reward penalty per agent detected on dry ground. Default: 500.0.
- shallow_penalty_per_agent (float): Reward penalty per agent in very shallow water. Default: 200.0.
- upstream_metric_window (int): Window length (timesteps) used for rolling averaging of upstream progress / velocity. Default: 30.
- too_shallow (float): Depth threshold (meters) below which water is considered too shallow for safe swimming (used for wet/dry detection).

UI ideas:
- Toggle `death_on_dry_immediate` and numeric field for `time_out_of_water_threshold`.
- Slider for `initial_velocity_jitter` and `initial_sog_jitter_fraction` plus a "Strong SOG jitter" checkbox.
- Numeric inputs for penalty magnitudes and upstream averaging window.
- Toggle which metrics to plot in the Metrics panel (collisions, mean upstream progress, mean upstream velocity, energy efficiency).

Notes:
- Defaults chosen to be conservative; aggressive settings (instant death, large penalties) can be used for training but may prematurely reduce exploration.
- Changes to these parameters can be made via the simulation object before launching the viewer, e.g.:

```
sim.death_on_dry_immediate = True
sim.initial_velocity_jitter = 0.1
rl_trainer.initial_sog_jitter_fraction = 0.2
```