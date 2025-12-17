"""Movement kernels package with runtime selection between numba and numpy implementations."""
from importlib import import_module

__all__ = [
    "swim_core",
    "drag_and_battery",
    "calc_battery",
    "merged_battery",
    "bout_distance",
    "time_to_fatigue",
]

def _try_import(module_name, attr=None):
    try:
        mod = import_module(module_name)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None

# Prefer numba modules if available
swim_core = _try_import("src.emergent.fish_passage.movement.swim_core_numba", "swim_core") or _try_import("src.emergent.fish_passage.movement.swim_core_numpy", "swim_core")
drag_and_battery = _try_import("src.emergent.fish_passage.movement.drag_and_battery_numba", "drag_and_battery") or _try_import("src.emergent.fish_passage.movement.drag_and_battery_numpy", "drag_and_battery")
calc_battery = _try_import("src.emergent.fish_passage.movement.calc_battery_numba", "calc_battery") or _try_import("src.emergent.fish_passage.movement.calc_battery_numpy", "calc_battery")
merged_battery = _try_import("src.emergent.fish_passage.movement.merged_battery_numba", "merged_battery") or _try_import("src.emergent.fish_passage.movement.merged_battery_numpy", "merged_battery")
bout_distance = _try_import("src.emergent.fish_passage.movement.bout_distance_numba", "bout_distance") or _try_import("src.emergent.fish_passage.movement.bout_distance_numpy", "bout_distance")
time_to_fatigue = _try_import("src.emergent.fish_passage.movement.time_to_fatigue_numba", "time_to_fatigue") or _try_import("src.emergent.fish_passage.movement.time_to_fatigue_numpy", "time_to_fatigue")

if swim_core is None:
    raise ImportError("No swim_core implementation available (numba or numpy)")
