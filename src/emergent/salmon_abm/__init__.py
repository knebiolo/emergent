# -*- coding: utf-8 -*-

"""Top-level exports for salmon_abm submodules.

We export the new modular implementations so callers can do:
	from emergent.salmon_abm import movement, behavior, fatigue

Legacy modules are imported in try/except blocks to avoid test-time
side-effects.
"""
__all__ = [
	'movement',
	'behavior',
	'fatigue',
	'simulation',
	'hdf5_io',
]

try:
	from emergent.salmon_abm import movement, behavior, fatigue, simulation, hdf5_io
except Exception:
	# Tests and minimal environments may import submodules directly.
	pass
