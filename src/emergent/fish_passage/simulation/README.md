Orchestrator
============

This module provides a simple `Orchestrator` to run short deterministic scenarios for testing and integration.

Usage:

```python
from emergent.fish_passage.simulation.orchestrator import Orchestrator
o = Orchestrator(n_agents=10, warmup_numba=True)
log = o.run(steps=100, dt=1.0)
```

Notes:
- Callers can add hooks via `add_hook(fn)` where `fn(t, out, pid_out)` will be invoked each timestep.
- Warmup is best-effort; callers may want to explicitly call `warmup.warmup_swim_core(...)` before benchmarking.
