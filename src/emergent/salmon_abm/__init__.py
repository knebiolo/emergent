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
	# configure package logging defaults (callers may override)
	try:
		from emergent.salmon_abm.logging_config import configure_logging
		configure_logging()
	except Exception:
		# non-fatal: proceed even if logging setup fails
		pass
except Exception:
	# Tests and minimal environments may import submodules directly.
	pass
