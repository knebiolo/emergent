PID tuning analysis — Zigzag long-run sweep
===========================================

Date: 2025-10-13

Overview
--------
This document summarizes the automated long-run zigzag experiments performed to evaluate and tune the simulation-level rudder PID controller. The goal was to remove short-run oscillations and find a conservative set of defaults that give small overshoot, small steady-state error, and stable behavior under spatially-varying wind forcing.

Test setup
----------
- Test: headless zigzag test (single vessel)
- Zigzag amplitude: ±10°
- Hold per leg: 60 s
- Total simulation time per run: 360 s
- Time step: 0.5 s
- Environmental forcing: winds and currents disabled for deterministic baseline; wind forcing scale set to 0.35 in the tuning runs (conservative reduction to aerodynamic forcing)
- PID trace CSVs written per run to `traces/`

Sweep performed
---------------
- Kd (derivative gain) ∈ {1.0, 2.0, 3.0}
- deriv_tau (derivative low-pass time constant) ∈ {0.5, 1.0, 2.0}

Key results (summary)
----------------------
- Kd=1.0:
  - peak_overshoot ≈ 10.97°
  - steady_state_error ≈ −4.48° (significant bias)
  - oscillation_period ≈ 125.2 s
- Kd=2.0:
  - peak_overshoot ≈ 4.66°
  - settling_time ≈ 32.5 s
  - steady_state_error ≈ −2.85°
  - oscillation_period ≈ 122.8 s
- Kd=3.0:
  - peak_overshoot ≈ 1.31°
  - settling_time ≈ 47.0 s
  - steady_state_error ≈ −0.245° (near-zero bias)
  - oscillation_period ≈ 122.0 s

Interpretation
--------------
- The derivative gain (Kd) is the dominant knob controlling overshoot and steady-state bias in the tested configuration. Raising Kd from 1.0 → 3.0 substantially reduced overshoot and nearly eliminated steady-state bias.

- The derivative low-pass time constant (`deriv_tau`) had minimal effect on the scalar metrics we extracted across the tested values (0.5–2.0 s). Visual inspection of the PID traces (recommended next step) may show differences in noise and short-timescale D-term behavior, but the global zigzag metrics were insensitive in this sweep.

- The long-run oscillation period (≈122 s) is tied to the zigzag hold timing (60 s per leg) rather than an inherent high-frequency instability. Earlier 30 s oscillations observed in short tests were likely caused by resonance with the short hold time used in those tests.

Recommendations (apply to config.py)
-----------------------------------
Based on the sweep, adopt the following conservative defaults to improve handling stability across scenarios:

- CONTROLLER_GAINS['Kd'] = 3.0
- ADVANCED_CONTROLLER['deriv_tau'] = 1.0  # smoother D-term
- ADVANCED_CONTROLLER['trim_band_deg'] = 2.0  # reduce micro-twitching under gusts
- SHIP_AERO_DEFAULTS['wind_force_scale'] = 0.35  # limit wind forcing until aero model validated

These are now applied to `src/emergent/ship_abm/config.py`.

Suggested follow-ups
--------------------
1. Generate diagnostic plots for the long-run and the best-case sweep runs (P/I/D traces, applied rudder, heading error). Save plots to `figs/` and include captions explaining the behavior.
2. Run a small grid sweep on Kp and Ki around the chosen Kd to ensure no residual bias or slow drift exists under more realistic wind/current fields.
3. If you enable the real OFS winds/currents, re-check the chosen defaults — aerodynamic scaling is conservative; as the aero model improves, wind_force_scale can be increased.
4. Consider integrating an automatic logger that records when the integrator is prevented from growing (saturation events) so you can quantitatively detect anti-windup activation.

Files produced during the tuning session
--------------------------------------
- `scripts/run_zigzag_long.py` — headless runner for long zigzag tests
- `scripts/zigzag_focus_sweep.py` — sweep driver (Kd × deriv_tau)
- `traces/` — per-run PID trace CSVs
- `zigzag_focus_summary.csv` — aggregated metrics for the sweep
- `pid_trace_zigzag_long.csv` — earlier long-run trace

Contact / Notes
---------------
If you want, I can now:
- produce the plots and add them to a `figs/` directory and a short report; or
- commit `config.py` changes to your repo branch and open a PR (if you want a formal commit history entry).



## Diagnostic plots
![Long run (360s) with conservative tuning (Kd=3.0)](figs\pid_trace_long.png)

*Long run (360s) with conservative tuning (Kd=3.0)* — see `figs\pid_trace_long.png`.

![Best-case sweep trace: pid_trace_kd3p0_tau0p5](figs\pid_trace_pid_trace_kd3p0_tau0p5.png)

*Best-case sweep trace: pid_trace_kd3p0_tau0p5* — see `figs\pid_trace_pid_trace_kd3p0_tau0p5.png`.

![Best-case sweep trace: pid_trace_kd3p0_tau1p0](figs\pid_trace_pid_trace_kd3p0_tau1p0.png)

*Best-case sweep trace: pid_trace_kd3p0_tau1p0* — see `figs\pid_trace_pid_trace_kd3p0_tau1p0.png`.

![Best-case sweep trace: pid_trace_kd3p0_tau2p0](figs\pid_trace_pid_trace_kd3p0_tau2p0.png)

*Best-case sweep trace: pid_trace_kd3p0_tau2p0* — see `figs\pid_trace_pid_trace_kd3p0_tau2p0.png`.

