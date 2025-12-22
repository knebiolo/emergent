"""Compatibility shim that re-exports legacy API names for consumers of `sockeye`.

Lightweight file that keeps old imports working while encouraging migration to
the new modules. Calling code should import from `emergent.salmon_abm` instead.
"""
import warnings

warnings.warn(
    "emergent.salmon_abm.compat is a temporary compatibility shim. Import from the new submodules instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    # Prefer new implementation
    from emergent.salmon_abm.simulation import simulation as simulation
except Exception:
    # fallback to legacy sockeye if present
    from emergent.salmon_abm import sockeye as _sockeye
    simulation = getattr(_sockeye, 'simulation', None)

try:
    from emergent.salmon_abm.pid import PID_controller as PID_controller
except Exception:
    from emergent.salmon_abm import sockeye as _sockeye
    PID_controller = getattr(_sockeye, 'PID_controller', None)

# summary: legacy summary object lives in sockeye; re-export if available
try:
    from emergent.salmon_abm.sockeye import summary as summary
except Exception:
    summary = None

__all__ = ['simulation', 'PID_controller', 'summary']
