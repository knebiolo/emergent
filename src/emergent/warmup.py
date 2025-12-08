"""Public warmup helpers for the emergent package.

This module exposes a stable API that tools and runners can call to
prime Numba kernels used by the simulation. It delegates to the
simulation-specific helper `_numba_warmup_for_sim` when available.
"""
from __future__ import annotations

import importlib
import types
from typing import Optional


def _get_sockeye_module() -> Optional[types.ModuleType]:
    # Prefer the canonical `sockeye` module.  Fall back to legacy names
    # for compatibility with older callers/scripts.
    candidates = [
        'emergent.salmon_abm.sockeye',
        'emergent.salmon_abm.sockeye_SoA',
        'emergent.salmon_abm.sockeye_SoA_OpenGL_RL',
        'emergent.salmon_abm.sockeye_SoA_OpenGL'
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def numba_warmup_for_sim(sim) -> None:
    """Call the simulation-specific warmup helper if available.

    This will call `_numba_warmup_for_sim(sim)` defined in the sockeye
    module if present. Tools should create a dummy object with `num_agents`
    and `swim_speeds` attributes and pass it here.
    """
    mod = _get_sockeye_module()
    if mod is None:
        raise RuntimeError('sockeye_SoA module not importable; cannot warmup')
    if hasattr(mod, '_numba_warmup_for_sim'):
        try:
            mod._numba_warmup_for_sim(sim)
            return
        except Exception as e:
            raise
    raise RuntimeError('sockeye_SoA._numba_warmup_for_sim not found')


def run_global_warmup(agent_count: int = 2000) -> None:
    """Create a minimal dummy sim and run numba warmup for common kernels.

    This is a convenience wrapper for tools that want to warm-up without
    constructing a full simulation object.
    """
    class _Dummy:
        pass

    d = _Dummy()
    d.num_agents = int(agent_count)
    import numpy as np
    d.swim_speeds = np.zeros((d.num_agents, 8), dtype=np.float64)
    numba_warmup_for_sim(d)
